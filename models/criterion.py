import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)
        return err


class SISNRLoss(BaseLoss):
    def __init__(self, eps=1e-6):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        assert targets.size() == preds.size()  # (B, L)
        # Step 1. Zero-mean norm
        targets = targets - torch.mean(targets, axis=-1, keepdim=True)
        preds = preds - torch.mean(preds, axis=-1, keepdim=True)
        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(targets ** 2, axis=-1, keepdim=True) + self.eps
        proj = torch.sum(targets * preds, axis=-1, keepdim=True) * targets / ref_energy
        # e_noise = s' - s_target
        noise = preds - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis=-1) / (torch.sum(noise ** 2, axis=-1) + self.eps)
        sisnr = 10 * torch.log10(ratio + self.eps)
        return -torch.mean(sisnr)
