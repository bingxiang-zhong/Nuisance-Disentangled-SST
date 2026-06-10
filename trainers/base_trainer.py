from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, Iterable, Optional
import torch
from tqdm import tqdm
from utils import build_warmup_cosine_scheduler


class BaseTrainer(ABC):
    """
    Base trainer class.

    Subclasses must define:
        - build_feature_extractor
        - build_model
        - build_loss
        - prepare_batch
        - compute_loss_and_metrics
    """

    def __init__(self, params: Dict[str, Any], training: bool = True):
        super().__init__()

        self.params = params
        self.training_mode = params['training_mode']
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_activated = self.dev.type == "cuda"
        self.use_amp = params["training"].get("use_amp", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
   
       
        

        # Build subclass-specific modules
        self.feature_extractor = self.build_feature_extractor(params)

        self.build_model(params)
        self.loss_fn = self.build_loss(params)

        checkpoint_path = params.get("model_checkpoint_path", "")
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.to(self.dev)
        if training:
            self.build_optimizer(params)
            self.build_lr_scheduler(
                params,
                warm_up_ratio=params.get("warm_up_ratio", 0.05),
                min_lr_ratio=params.get("min_lr_ratio", 10)
            )

    # ============================================================
    # Abstract API
    # ============================================================
    @abstractmethod
    def build_feature_extractor(self, params):
        pass

    @abstractmethod
    def build_model(self, params):
        pass

    @abstractmethod
    def build_optimizer(self, params):
        pass

    @abstractmethod
    def build_loss(self, params):
        pass

    @abstractmethod
    def modules_for_device(self):
        """
        Return a dict of modules that should be moved to device.
        """
        pass

    @abstractmethod
    def forward_base(self, inputs, targets, epoch=None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def forward_disentangled(self, inputs, targets) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        pass

    @abstractmethod
    def prepare_batch(self, mic_sig_batch, targets_batch=None):
        """
        Return:
            inputs: model input
            targets: target dict or None
        """
        pass

    @abstractmethod
    def compute_loss_and_metrics(self, inputs, targets=None, training: bool = True):
        """
        Return:
            loss: torch.Tensor
            metrics: dict[str, float]
        """
        pass

    @abstractmethod
    def get_trainable_modules(self) -> Dict[str, Optional[torch.nn.Module]]:
        """
        Example:
        {
            "model": self.model,
            "feature_extractor": self.feature_extractor,
            "FE": self.FE,
            "Disentangler_doa": self.Disentangler_doa,
            ...
        }
        """
        pass

    def modules_for_mode(self) -> Dict[str, torch.nn.Module]:
        all_modules = self.get_trainable_modules()

        if self.training_mode == "base":
            keys = ["feature_extractor", "model"]
        elif self.training_mode == "disentangled":
            keys = [
                "feature_extractor",
                "FE",
                "Disentangler_doa",
                "Disentangler_nui",
                "Localizer",
                "CLUBLoss",
                
            ]
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        out = {}
        for k in keys:
            m = all_modules.get(k, None)
            if m is not None:
                out[k] = m
        return out


    def set_mode(self, training: bool = True):
        for _, module in self.modules_for_mode().items():
            if training:
                module.train()
            else:
                module.eval()

    def set_train_mode(self):
        self.set_mode(training=True)

    def set_test_mode(self):
        self.set_mode(training=False)

    @staticmethod
    def set_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad

    def set_requires_grad_by_names(self, module_names: Iterable[str], requires_grad: bool):
        all_modules = self.get_trainable_modules()
        for name in module_names:
            self.set_requires_grad(all_modules.get(name, None), requires_grad)

    @contextmanager
    def freeze_modules(self, module_names: Iterable[str]):
        self.set_requires_grad_by_names(module_names, False)
        try:
            yield
        finally:
            self.set_requires_grad_by_names(module_names, True)


    # ============================================================
    # Optimizer / Scheduler
    # ============================================================


    def build_lr_scheduler(self, params, warm_up_ratio=0.05, min_lr_ratio=10):
        model_name = params["model"]

        base_lr = params["training"]["lr"]
        nb_epoch = params["training"]["nb_epochs"]
        steps_per_epoch = params["len_train_loader"]
        total_steps = nb_epoch * steps_per_epoch

        warmup_steps = int(warm_up_ratio * total_steps)
        eta_min = base_lr / min_lr_ratio
      
        self.lr_scheduler = build_warmup_cosine_scheduler(
                optimizer=self.optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                base_lr=base_lr,
                eta_min=eta_min,
            )

    # ============================================================
    # Device
    # ============================================================
    def to(self, device):
        self.dev = torch.device(device) if not isinstance(device, torch.device) else device
        self.cuda_activated = self.dev.type == "cuda"

        # if self.model is not None:
        #     self.model.to(self.dev)
        for _, module in self.modules_for_device().items():
            if module is not None:
                module.to(self.dev)

        if self.feature_extractor is not None:
            self.feature_extractor.to(self.dev)

        if isinstance(self.loss_fn, torch.nn.Module):
            self.loss_fn.to(self.dev)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def cpu(self):
        return self.to(torch.device("cpu"))

    def _align_dimensions(self, pred_batch, targets):
        doa = targets["doa"]
        vad = targets["vad"]

        if pred_batch.shape[1] > doa.shape[1]:
            pred_batch = pred_batch[:, :doa.shape[1], :]
        else:
            doa = doa[:, :pred_batch.shape[1], :]
            vad = vad[:, :pred_batch.shape[1], :]

        aligned_targets = {
            "doa": doa,
            "vad": vad,
        }
        return pred_batch, aligned_targets

    def train_step_base(self, inputs, targets, epoch=None):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.forward_base(inputs, targets, epoch=epoch)
            loss_dict, _ = self.compute_loss_and_metrics(outputs, targets, training=True)
            total_loss = loss_dict["doa_loss"]
            


        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        
        self.lr_scheduler.step()

        log_msg = {
            "doa_loss": total_loss.item(),
        }

        return log_msg

    def get_lambda_mi(self, epoch=None):
        if not hasattr(self, "lambda_mi"):
            return 0.0

        if epoch is None:
            return self.lambda_mi

        warmup_epochs = getattr(self, "lambda_mi_warmup_epochs", 5)
        ratio = min(1.0, float(epoch) / max(1, warmup_epochs))
        return ratio * self.lambda_mi


    def train_step_disentangled(self, inputs, targets, epoch=None):
        # ============================================================
        # phase 1: update CLUB estimator only
        # ============================================================
        with self.freeze_modules(["FE", "Disentangler_doa", "Disentangler_nui"]):
            self.optimizer_club.zero_grad()

            with torch.no_grad():
                outputs_club = self.forward_disentangled(inputs, targets)

            feat_doa = outputs_club["feat_doa"]
            feat_nui = outputs_club["feat_nui"]
         
            _, targets = self._align_dimensions(outputs_club['pred_doa'], targets)
           
            club_learning_loss = self.CLUBLoss.learning_loss(feat_doa, feat_nui)
            club_learning_loss.backward()
            self.optimizer_club.step()

        # ============================================================
        # phase 2: update main network, freeze CLUB
        # ============================================================
        with self.freeze_modules(["CLUBLoss"]):
            self.optimizer.zero_grad()

            outputs = self.forward_disentangled(inputs, targets)
            loss_dict, _ = self.compute_loss_and_metrics(outputs, targets, training=True)
            _, targets = self._align_dimensions(outputs['pred_doa'], targets)
            vad = targets['vad']
            club_loss = self.CLUBLoss(outputs["feat_doa"], outputs["feat_nui"])
            lambda_mi = self.get_lambda_mi(epoch)

            total_loss = (
                loss_dict["doa_loss"]
                + lambda_mi * club_loss
                + 1e-3 * loss_dict["nui_loss"]
               
            )

            total_loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log_msg = {
            "doa_loss": float(loss_dict["doa_loss"].item()),
            "nui_loss": float(loss_dict["nui_loss"].item()),
            "club_loss": float(club_loss.item()),
            "total_loss": float(total_loss.item()),
        }
       
        return log_msg


    def train_epoch(self, dataloader, epoch=None):
        self.set_train_mode()

        pbar = tqdm(dataloader, ascii=True, desc=f"Epoch {epoch}")
        for mic_sig_batch, targets_batch in pbar:
            inputs, targets = self.prepare_batch(mic_sig_batch, targets_batch)

            if self.training_mode == "base" :
                log_msg = self.train_step_base(inputs, targets, epoch=epoch)
          
              
            elif self.training_mode == "disentangled":
                log_msg = self.train_step_disentangled(inputs, targets, epoch=epoch)
            else:
                raise ValueError(f"Unknown training_mode: {self.training_mode}")
                

            pbar.set_postfix(**log_msg)



    def test_epoch(self, dataloader):
        self.set_test_mode()
        total_samples = 0
        total_metrics = {}

        with torch.no_grad():
            for mic_sig_batch, targets_batch in tqdm(dataloader, ascii=True):
                inputs, targets = self.prepare_batch(mic_sig_batch, targets_batch)
                if self.training_mode == "base":
                    outputs = self.forward_base(inputs, targets)
                elif self.training_mode == "disentangled":
                    outputs = self.forward_disentangled(inputs, targets)

                loss, batch_metrics = self.compute_loss_and_metrics(outputs, targets, training=False)

                batch_size = targets["doa"].shape[0]
                total_samples += batch_size

                for key, value in {"loss": loss['doa_loss'], **batch_metrics}.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()

                    total_metrics[key] = total_metrics.get(key, 0.0) + float(value) * batch_size

        if total_samples == 0:
            return {}

        return {key: value / total_samples for key, value in total_metrics.items()}
