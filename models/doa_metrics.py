import math
import torch
import torch.nn as nn


def angular_error_deg(pred, target):
    diff = pred - target
    diff = (diff + 180.0) % 360.0 - 180.0
    return diff.abs()


class DoaEvaluator(nn.Module):
    def __init__(self, dev, ae_th=5.0, use_vad=True, vad_th=(0.001, 0.05)):
        super().__init__()
        self.dev = dev
        self.ae_th = ae_th
        self.use_vad = use_vad
        self.vad_th = vad_th

    def forward(self, pred_batch, gt_batch, tar_type="spect"):
        doa_gt = self._to_bt(gt_batch[0]).to(self.dev)
        vad_gt = self._to_bt(gt_batch[1]).to(self.dev)

        if tar_type == "spect":
            vad_est, doa_est = pred_batch.topk(k=1, dim=-1)   # (B, T, 1), (B, T, 1)
            doa_est = self._to_bt(doa_est).to(self.dev)
            vad_est = self._to_bt(vad_est).to(self.dev)

        elif tar_type == "degree":
            doa_est = self._to_bt(pred_batch).to(self.dev) 
            vad_est = torch.ones_like(vad_gt)

        else:
            raise ValueError(f"Unsupported tar_type: {tar_type}")

        if self.use_vad:
            active = (vad_gt > self.vad_th[0]).float()
        else:
            active = torch.ones_like(vad_gt, dtype=torch.float32)

        azi_error = angular_error_deg(doa_est, doa_gt)

        correct = (azi_error < self.ae_th).float() * active

        denom = active.sum().clamp_min(1.0)
        acc = correct.sum() / denom
        mae = (azi_error * active).sum() / denom

        denom_batch = active.sum(dim=1).clamp_min(1.0)
        mae_batch = (azi_error * active).sum(dim=1) / denom_batch

        mae_frame_active = azi_error[active.bool()]

        return {
            "ACC": acc,
            "MAE": mae,
            "MAE_batch": mae_batch,
            "MAE_frame": mae_frame_active,
            "MAE_frame_all": azi_error, 
            "active_mask": active.bool(),
        }

    @staticmethod
    def _to_bt(x):
        if x.ndim == 2:
            return x
        if x.ndim == 3 and x.shape[-1] == 1:
            return x.squeeze(-1)
        raise ValueError(f"Expected shape (B, T) or (B, T, 1), got {x.shape}")