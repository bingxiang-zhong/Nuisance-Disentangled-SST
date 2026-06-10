import numpy as np
import torch
import torch.nn.functional as F
from models.doa_metrics import DoaEvaluator
import models.module as at_module
from models.crnn_disentangle import CRNN, crnnFE, Disentangler, CLUBLoss, Localizer

from trainers.base_trainer import BaseTrainer
import torch.nn as nn

class CRNNTrainer(BaseTrainer):
    """
    Trainer for CRNN model using SRP-like spectral targets.
    """

    def __init__(self, params, training: bool = True):
        self.res_phi = params["res_phi"]
        super().__init__(params, training=training)
        self.lambda_mi = params["training"].get("lambda_mi", 1e-5)
        self.lambda_mi_warmup_epochs = params["training"].get("lambda_mi_warmup_epochs", 5)
        self.gaussian_sigma = params['sigma']
        self.get_metric = DoaEvaluator(self.dev)
        self.gaussian_lut = self.build_gaussian_lut(self.res_phi, sigma=self.gaussian_sigma).to(self.dev)
        self.mic_num = params["mic_num"]

    # ============================================================
    # Build subclass-specific modules
    # ============================================================
    def build_feature_extractor(self, params):
        return CRNNFeatureExtractor(params)

    def get_trainable_modules(self):
        return {
            "model": getattr(self, "model", None),
            "feature_extractor": getattr(self, "feature_extractor", None),
            "FE": getattr(self, "FE", None),
            "Disentangler_doa": getattr(self, "Disentangler_doa", None),
            "Disentangler_nui": getattr(self, "Disentangler_nui", None),
            "Localizer": getattr(self, "Localizer", None),
            "CLUBLoss": getattr(self, "CLUBLoss", None),
        }
    
    def build_model(self, params):
        
        if self.training_mode == "base":
            self.model = CRNN(params, cnn_in_dim=params["mic_num"] * 2, cnn_dim=64, res_Phi=self.res_phi)
           
            self.FE = None
            self.Disentangler_doa = None
            self.Disentangler_nui = None
            self.Localizer = None
            self.CLUBLoss = None

        elif self.training_mode == "disentangled":
            self.model = None

            self.FE = crnnFE(params)
            self.Disentangler_doa = Disentangler()
            self.Disentangler_nui = Disentangler()
            self.CLUBLoss = CLUBLoss(
                x_dim=256,
                y_dim=256,
                hidden_size=128,
            )
            self.Localizer = Localizer(res_Phi=self.res_phi)

        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")


    def build_loss(self, params):
        return torch.nn.KLDivLoss(reduction="batchmean")

    def build_optimizer(self, params):
        lr = params["training"]["lr"]

        if self.training_mode == "base":
            trainable_params = list(self.model.parameters())
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
            self.optimizer_club = None

        elif self.training_mode == "disentangled":
            parameters = [
                {"params": self.FE.parameters(), "lr": lr},
                {"params": self.Disentangler_doa.parameters(), "lr": lr},
                {"params": self.Disentangler_nui.parameters(), "lr": lr},
                {"params": self.Localizer.parameters(), "lr": lr},
            ]

            self.optimizer = torch.optim.Adam(parameters, lr=lr)
            self.optimizer_club = torch.optim.Adam(
                self.CLUBLoss.parameters(),
                lr=params.get("club_lr", 1e-4),
            )

    def modules_for_device(self):
        modules = {
            "feature_extractor": self.feature_extractor,
            "model": self.model,
            "FE": self.FE,
            "Disentangler_doa": self.Disentangler_doa,
            "Disentangler_nui": self.Disentangler_nui,
            "CLUBLoss": self.CLUBLoss,
            "Localizer": self.Localizer,
         
        }

        if isinstance(self.loss_fn, nn.Module):
            modules["loss_fn"] = self.loss_fn
        elif isinstance(self.loss_fn, dict):
            for k, v in self.loss_fn.items():
                if isinstance(v, nn.Module):
                    modules[f"loss_{k}"] = v

        return modules

    def save_checkpoint(self, path):
        print(f"Saving checkpoint to {path}")

        if self.training_mode == "base":
            torch.save(self.model.state_dict(), path)
        elif self.training_mode == "disentangled":
            torch.save(
                {
                    "training_mode": "disentangled",
                    "FE": self.FE.state_dict(),
                    "disentangler_doa": self.Disentangler_doa.state_dict(),
                    "disentangler_nui": self.Disentangler_nui.state_dict(),
                    "localizer": self.Localizer.state_dict(),
                },
                path,
            )

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location="cpu")

        if self.training_mode == "base":
            self.model.load_state_dict(state)
            
        elif self.training_mode == "disentangled":
            self.FE.load_state_dict(state["FE"])
            self.Disentangler_doa.load_state_dict(state["disentangler_doa"])
            self.Disentangler_nui.load_state_dict(state["disentangler_nui"])
            self.Localizer.load_state_dict(state["localizer"])

    # ============================================================
    # Batch preparation
    # ============================================================
    def prepare_batch(self, mic_sig_batch, targets_batch=None):
        mic_sig_batch = mic_sig_batch.to(self.dev, non_blocking=True)
        inputs = self.feature_extractor(mic_sig_batch)

        targets = None
        if targets_batch is not None:
            targets = {
                "doa": targets_batch["doa"].to(self.dev, non_blocking=True),
                "vad": targets_batch["vad"].to(self.dev, non_blocking=True),
            }

        return inputs, targets


    def forward_base(self, inputs, targets=None, epoch=None):
        outputs = self.model(inputs)
        return {"pred_doa": outputs}

    def forward_disentangled(self, inputs, targets, epoch=None):

          feat_state = self.FE(inputs)
          feat_doa = self.Disentangler_doa(feat_state)
          feat_nui = self.Disentangler_nui(feat_state)

          pred_doa = self.Localizer(feat_doa)

          
          with self.freeze_modules(["Localizer"]):
              pred_nui = self.Localizer(feat_nui)

          return {
              "feat_state": feat_state,
              "feat_doa": feat_doa,
              "feat_nui": feat_nui,
              "pred_doa": pred_doa,
              "pred_nui": pred_nui,
          }


    

    def _compute_uniform_loss(self, logits: torch.Tensor, eps: float = 1e-8):

        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()

        n_class = logits.size(-1)
        uniform = torch.full_like(p, 1.0 / n_class)

        # KL(U || P)
        loss = (uniform * (torch.log(uniform.clamp_min(eps)) - logp)).sum(dim=-1)
        return loss.mean()
    


    def compute_loss_and_metrics(self, outputs, targets=None, training=True):
        loss_dict = {}
        metrics = {}


        if self.training_mode == "base":
            pred_doa, targets = self._align_dimensions(outputs['pred_doa'], targets)
   
            doa_loss = self._compute_gaussian_loss(pred_doa, targets)

            loss_dict["doa_loss"] = doa_loss
            

        elif self.training_mode == "disentangled":
            pred_doa, targets = self._align_dimensions(outputs['pred_doa'], targets)
            doa_loss = self._compute_gaussian_loss(pred_batch=pred_doa, targets=targets)
            loss_dict["doa_loss"] = doa_loss

            pred_nui, targets = self._align_dimensions(outputs['pred_nui'], targets)
            nui_loss = self._compute_gaussian_loss(pred_batch=pred_nui, targets=None)

            loss_dict["nui_loss"] = nui_loss


        if not training:
            gt_batch = [targets["doa"], targets["vad"]]
            metric = self.get_metric(
                pred_batch=pred_doa,
                gt_batch=gt_batch,
                tar_type="spect",
            )
        
            metrics = {
                "ACC": metric['ACC'].item(),
                "MAE": metric['MAE'].item(),
            }

        return loss_dict, metrics



    

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


    
    def _compute_gaussian_loss(self, pred_batch, targets=None):
        nb, nt, _ = pred_batch.shape
        if targets is not None:
            doa_batch = targets["doa"].long().to(self.dev)      # (nb, nt, max_sources)
            vad_batch = targets["vad"].to(self.dev).squeeze(-1)     # (nb, nt)

            gaussian_targets = self.gaussian_encode_symmetric_vectorized(
                doa_batch, self.res_phi)
            gaussian_targets = F.softmax(gaussian_targets, dim=-1)

            uniform_dist = torch.ones(self.res_phi, device=self.dev) / self.res_phi

            new_target_batch = torch.where(
                vad_batch.unsqueeze(-1),
                gaussian_targets,
                uniform_dist,
            )
        else:
            new_target_batch = torch.ones(
                nb, nt, self.res_phi, device=self.dev
            ) / self.res_phi

        pred_batch_logprob = F.log_softmax(pred_batch, dim=-1)

        loss = self.loss_fn(pred_batch_logprob, new_target_batch)
        return loss


    # ============================================================
    # Gaussian encoding
    # ============================================================
    def build_gaussian_lut(self, res_phi, sigma=8):
        angles = torch.arange(res_phi, device=self.dev).float()
        grid = angles.view(1, res_phi)
        centers = angles.view(res_phi, 1)

        diff = torch.abs(grid - centers)
        distance = torch.minimum(diff, 360 - diff)

        lut = torch.exp(-0.5 * (distance ** 2) / (sigma ** 2))
        return lut

    def gaussian_encode_symmetric_vectorized(self, angles, res_phi):
        nb, nt, max_sources = angles.shape

        angles = angles.clamp(0, res_phi - 1)
        gaussian = self.gaussian_lut[angles]      # (nb, nt, max_sources, res_phi)
        spectrum = gaussian.max(dim=2).values     # (nb, nt, res_phi)

        return spectrum


class CRNNFeatureExtractor(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        win_size = params["win_size"]
        hop_rate = params["hop_rate"]

        self.c = params["speed_of_sound"]
        self.fs = params["fs"]

        self.nfft = win_size
        self.res_phi = params["res_phi"]
        self.dostft = at_module.STFT(
            win_len=win_size,
            win_shift_ratio=hop_rate,
            nfft=win_size,
        )
        self.fre_range_used = range(1, int(self.nfft / 2) + 1, 1)

    def forward(self, mic_sig_batch, eps=1e-6):
        stft = self.dostft(signal=mic_sig_batch)
        nb, nf, nt, nc = stft.shape

        stft = stft.permute(0, 3, 1, 2)  # (B, C, F, T)

        mag = torch.abs(stft)
        mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
        mean_value = mean_value[:, None, None, None].expand_as(mag)

        stft_real = torch.real(stft) / (mean_value + eps)
        stft_imag = torch.imag(stft) / (mean_value + eps)

        real_imag_batch = torch.cat((stft_real, stft_imag), dim=1)
        data = real_imag_batch[:, :, self.fre_range_used, :]

        return data
