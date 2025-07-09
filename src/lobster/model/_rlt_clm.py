import importlib.resources
from typing import Callable, Literal, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler, pipeline

from lobster.tokenization import AminoAcidTokenizerFast
from lobster.transforms._patching_transform import PatchifierTransform
from lobster.transforms._patcher import Patcher, PatcherArgs

from .lm_base._latent_transformer import ByteLatentTransformer
from ._clm_configuration import PCLM_CONFIG_ARGS


class ResidueLatentTransformerCLM(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "rlt_clm_mini",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1000,
        config: dict = None,
        ckpt_path: str = None,
        max_seq_length: int = 2048,
        max_num_patches: int = 512,
        scheduler: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ] = "constant_with_warmup",
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
    ):
        """
        Prescient Protein Causal Language Model.

        Parameters
        ----------
        model_name: one of PCLM_CONFIG_ARGS.keys()
        num_key_value_heads: This is the number of key_value heads that should be used to implement
            Grouped Query Attention. If`num_key_value_heads=num_attention_heads`, the model will
            use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use
            Multi Query Attention (MQA) otherwise GQA is used.
        scheduler: str, optional
            The type of learning rate scheduler.
        model_kwargs: dict, optional
            Additional keyword arguments to pass to the model.
        scheduler_kwargs: dict, optional
            Additional keyword arguments to pass to the scheduler.

        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._ckpt_path = ckpt_path

        self._max_seq_length = max_seq_length
        self._max_num_patches = max_num_patches
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.model = ByteLatentTransformer(config)

        self.tokenizer = AminoAcidTokenizerFast()
        self._patchifier_transform = PatchifierTransform(
            AminoAcidTokenizerFast(),
            max_num_patches=self.max_num_patches,
            truncation=True,
            padding="max_length",
            random_truncation=True,
            max_seq_length=self._max_seq_length,
        )
        self.patcher = Patcher(PatcherArgs(
                patch_size= config.patch_size,
                patching_mode=config.patching_mode,
                threshold=config.patching_threshold,
                threshold_add=config.patching_threshold_add,
                monotonicity=config.monotonicity,
                max_patch_length=config.max_patch_length,
            )
        )

        self.config = config
        self.loss_fn = CrossEntropyLoss()

        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)

        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )

        # Create base kwargs for the scheduler
        scheduler_params = {
            "num_warmup_steps": self._num_warmup_steps,
            "num_training_steps": self._num_training_steps,
        }

        # Add any additional scheduler kwargs from initialization
        scheduler_params.update(self.scheduler_kwargs)

        scheduler = get_scheduler(self.scheduler, optimizer, **scheduler_params)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_loss(self, batch):
        output = self.model(
            tokens=batch['input_ids'].squeeze(1),
            patch_lengths=batch['patch_lengths'].squeeze(1),
            attn_mask=batch['attention_mask'],
            attn_patch_mask=batch['attention_mask_patch'],
            )
        loss = self.loss_fn(
            output.view(-1, self.config.vocab_size),  # [batch_size * seq_len, vocab_size]
            batch['labels'].view(-1)      # [batch_size * seq_len]
        )
 
        return loss
        return loss, logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        ONLY SUPPORTS SINGLE SEQUENCE INFERENCE!

        """
        attn_mask = input_ids != self.tokenizer.pad_token_id
        patch_lengths, _ = self.patcher.patch(
            input_ids,
            include_next_token=True, #CHECK FOR THIS
            threshold=self.patcher.threshold,
        )
        attn_patch_mask = patch_lengths != 0
        output = self.model(
            tokens=input_ids,
            attn_mask=attn_mask,
            patch_lengths=patch_lengths,
            attn_patch_mask=attn_patch_mask,
            )
        return output

    def sequences_to_log_likelihoods(self, sequences: list[str]) -> torch.Tensor:
        outputs = [self.get_nll_and_logits(s) for s in sequences]
        nlls = torch.stack([o[0] for o in outputs])
        return -nlls

    def get_nll_and_logits(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.as_tensor(self.tokenizer.encode(sequence), device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(input_ids)
            nll = nn.functional.cross_entropy(logits[:, :-1, :].flatten(end_dim=-2), input_ids[:, 1:].flatten())
        return nll, logits

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
