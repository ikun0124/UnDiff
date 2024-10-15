"""Loss function for the UnDiff Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn


class UnDiffLoss(nn.Module):
    """Uncertainty loss for the UnDiff model.
    """

    def __init__(self, wrec=50, wnig=1, wpenal=0.001, warmup=0):
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_rec = nn.L1Loss()

        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

        self.wrec = wrec
        self.wnig = wnig
        self.wpenal = wpenal

    def forward(
        self, latent_in: Tensor, latent_out: Tensor, images_input: Tensor, images_recon: Tensor, evidential_para: list, epoch_num: int) -> Tensor:
        """Compute the loss for a batch.

        Returns:
            Tensor: The computed uncertainty loss.
        """

        v_batch, alpha_batch, beta_batch = evidential_para[0] , evidential_para[1], evidential_para[2]
        loss_NIG = 0
        for i in range(latent_in.shape[0]):
            v, alpha, beta = v_batch[i], alpha_batch[i], beta_batch[i]
            loss_enc_mse = self.loss_mse(latent_in[i, :, :, :], latent_out[i, :, :, :])

            om = 2 * beta * (1 + v)
            loss_nll = (0.5 * torch.log(torch.pi / v + 1e-16) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(v * loss_enc_mse + om) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)).sum()
            loss_reg = (loss_enc_mse * (2 * v + alpha)).sum()
            loss_NIG += loss_nll + loss_reg * self.wpenal

        loss_NIG /= latent_in.shape[0]
        loss_rec = self.loss_l1(images_input, images_recon)
        loss_total = loss_NIG * self.wnig + loss_rec * self.wrec
        return loss_total
