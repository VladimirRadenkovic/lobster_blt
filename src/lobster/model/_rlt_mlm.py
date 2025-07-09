import copy
import importlib.resources
import os
from typing import Iterable, Literal, Optional, Union

import lightning.pytorch as pl
import pandas as pd
import torch
from transformers import get_scheduler
from transformers.configuration_utils import PretrainedConfig

from lobster.tokenization import AminoAcidTokenizerFast
from lobster.transforms._patching_transform import PatchifierTransform
from .lm_base._latent_transformer import ByteLatentTransformer

from lobster.transforms._patcher import Patcher, PatcherArgs

from torch.nn import CrossEntropyLoss


class ResidueLatentTransformerMLM(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "rlt_mlm_mini",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        freeze: bool = False,
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
        scheduler_kwargs: dict = None,
        model_type: str = "mlm",
    ):
        """
        MLM Residue Latent Transformer.
        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        self._freeze = freeze
        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._ckpt_path = ckpt_path
        self._max_seq_length = max_seq_length
        self._max_num_patches = max_num_patches
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.model_type = model_type
        self.model = ByteLatentTransformer(config)
        if self._freeze:
            self._freeze_all_but_lm_head()

        self.save_hyperparameters(logger=False)

        self.tokenizer = AminoAcidTokenizerFast()
        self._patchifier_transform = PatchifierTransform(
            AminoAcidTokenizerFast(),
            max_num_patches=self._max_num_patches,
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

    def _compute_loss(self, batch):
        output = self.model(
            tokens=batch['input_ids'].squeeze(1),
            patch_lengths=batch['patch_lengths'].squeeze(1),
            attn_mask=batch['attention_mask'],
            attn_patch_mask=batch['attention_mask_patch'],
            )
        loss = self.loss_fn(
            output.view(-1, self.config.get("vocab_size", 34)),  # [batch_size * seq_len, vocab_size]
            batch['labels'].view(-1)      # [batch_size * seq_len]
        )
 
        return loss

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ONLY SUPPORTS SINGLE SEQUENCE INFERENCE!

        """
        attn_mask = input_ids != self.tokenizer.pad_token_id
        patch_lengths, _ = self.patcher.patch(
            tokens=input_ids,
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )

        scheduler = get_scheduler(
            name=self.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
        with torch.inference_mode():
            hidden_states = self.model(
                tokens=batch['input_ids'].squeeze(1).to(self.device),
                patch_lengths=batch['patch_lengths'].squeeze(1).to(self.device),
                attn_mask=batch['attention_mask'].to(self.device),
                attn_patch_mask=batch['attention_mask_patch'].to(self.device),
            )
        # mean pool over AAs
        mean_embeddings = hidden_states.mean(dim=1).cpu().numpy()
        columns = [f"embedding_{idx}" for idx in range(hidden_states.shape[-1])]
        df = pd.DataFrame(
            mean_embeddings,
            columns=pd.Index(columns),
        )

        return df


    def naturalness(self, sequences: Iterable[str]) -> torch.Tensor:
        out = [
            self._naturalness_single_sequence(
                seq,
                batch_size=32,
            )
            for seq in sequences
        ]

        return torch.tensor(out)

    def _naturalness_single_sequence(
        self,
        sequence: str,
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> Union[float, tuple[float, Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        N = len(sequence)
        encoded_seq = torch.as_tensor(self.tokenizer.encode(sequence))
        mask_condition = torch.eye(encoded_seq.shape[0], dtype=torch.bool)
        seqs_mask_encoded = torch.where(mask_condition, self.tokenizer.mask_token_id, encoded_seq)
        ref_seq_indices = torch.tensor(encoded_seq) - 5

        if N < batch_size:
            batch_size_ = N
        else:
            batch_size_ = batch_size
        with torch.inference_mode():
            logits = torch.vstack(
                [
                    self.model(tokens=toks.to(self.device))
                    for toks in torch.tensor_split(seqs_mask_encoded, N // batch_size_)
                ]
            )

        # raw_log_probs [N, 20]: log probability for each WT amino acid
        raw_log_probs = torch.nn.functional.log_softmax(logits[:, 1:-1, 5:26], dim=-1)[
            torch.arange(N), torch.arange(N), :
        ]
        # sum of log probabilities that the model assigns to the true amino acid in each masked position
        sum_log_probs = raw_log_probs[torch.arange(N), ref_seq_indices[1:-1]].sum()  # chop off bos/eos

        naturalness_score = (1.0 / torch.exp(-sum_log_probs / N)).item()

        if return_probs:
            return naturalness_score, (raw_log_probs, ref_seq_indices[1:-1].detach())
        else:
            return naturalness_score

    def _freeze_all_but_lm_head(self):
        for name, param in self.model.named_parameters():
            if "output" not in name:  # Skip the output layer
                param.requires_grad = False

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
