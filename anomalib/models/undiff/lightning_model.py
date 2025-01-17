""" UnDiff Model
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule
from anomalib.models.undiff.loss import  UnDiffLoss

from .torch_model import UnDiffModel

logger = logging.getLogger(__name__)


class UnDiff(AnomalyModule):
    """PL Lightning Module for the UnDiff Algorithm.

    Args:
        batch_size (int): Batch size.
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        wrec (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        n_features: int,
        latent_vec_size: int,
        num_input_channels: int,
        wrec: int = 50,
        wnig: int = 1,
        wpenal: float = 0.001,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        warmup: int = 0,
    ) -> None:
        super().__init__()

        self.model: UnDiffModel = UnDiffModel(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
        )

        self.min_scores: Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.loss = UnDiff(wrec, wnig, wpenal, warmup)

        # TODO: LR should be part of optimizer in config.yaml! Since UnDiff has custom
        #   optimizer this is to be addressed later.
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epoch_num = 0

    def _reset_min_max(self) -> None:
        """Resets min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_g = optim.Adam(
            self.model.undiff.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        return  optimizer_g

    def training_step(
        self, batch: dict[str, str | Tensor], batch_idx: int
    ) -> STEP_OUTPUT:  # pylint: disable=arguments-differ
        """Training step.

        Args:
            batch (dict[str, str | Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.

        # forward pass
        images_input, images_recon, latent_in, latent_out, [v, alpha, beta] = self.model(batch["data"])
        loss = self.loss(latent_in, latent_out, images_input, images_recon, [v, alpha, beta], self.epoch_num)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        self.epoch_num += 1

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        batch["pred_scores"] = self.model(batch["data"])
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))

        return batch

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing validation outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().validation_epoch_end(outputs)
        return outputs

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step."""
        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing test outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().test_epoch_end(outputs)
        return outputs

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        scores = (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
        return scores


class UnDiffLightning(UnDiff):
    """PL Lightning Module for the UnDiff Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            batch_size=hparams.dataset.train_batch_size,
            num_input_channels=hparams.dataset.flow_num,
            input_size=hparams.model.input_size,
            n_features=hparams.model.n_features,
            latent_vec_size=hparams.model.latent_vec_size,
            wrec=hparams.model.wrec,
            wnig=hparams.model.wnig,
            wpenal=hparams.model.wpenal,
            lr=hparams.model.lr,
            beta1=hparams.model.beta1,
            beta2=hparams.model.beta2,
            warmup=hparams.model.warmup,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[Callback]:
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]