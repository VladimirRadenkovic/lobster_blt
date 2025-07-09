from typing import List, Optional, Union
import torch
from torch.nn import Module
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy

from lobster.transforms._patcher import Patcher, PatcherArgs
"""

class PatchifierTransform(Module):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        random_truncation: bool = False,
        max_seq_length: Optional[int] = None,
        max_num_patches: int = 512,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        patch_size: int = 4,
        patching_mode: str = "static",
        threshold: float = 0.0,
        threshold_add: float = 0.0,
        monotonicity: bool = False,
        seed: int = 42,
        model_type: str = "mlm",
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self._padding = padding
        self._truncation = truncation
        self.random_truncation = random_truncation
        self._max_seq_length = max_seq_length
        self._return_token_type_ids = return_token_type_ids
        self._return_attention_mask = return_attention_mask
        self._return_overflowing_tokens = return_overflowing_tokens
        self._return_special_tokens_mask = return_special_tokens_mask
        self._return_offsets_mapping = return_offsets_mapping
        self._return_length = return_length
        self._verbose = verbose

        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.threshold = threshold
        self.threshold_add = threshold_add
        self.monotonicity = monotonicity
        self.max_num_patches = max_num_patches
        self.seed = seed
        self.patches_args = PatcherArgs(
            patch_size=self.patch_size,
            patching_mode=self.patching_mode,
            threshold=self.threshold,
            threshold_add=self.threshold_add,
            monotonicity=self.monotonicity,
            max_patch_length=self.max_num_patches,
        )

        self.patcher = Patcher(
            patcher_args=self.patches_args
        )

        self.rand_generator = torch.Generator().manual_seed(self.seed)
        self.model_type = model_type

    def forward(
        self,
        text: Union[str, List[str], List[int]],
    ) -> BatchEncoding:
        batchEncoding = self.tokenizer(
            text,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_seq_length,
            return_tensors="pt",
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=self._return_overflowing_tokens,
            return_special_tokens_mask=self._return_special_tokens_mask,
            return_offsets_mapping=self._return_offsets_mapping,
            return_length=self._return_length,
            verbose=self._verbose,
        )

        tokens = batchEncoding["input_ids"]
        attention_mask = batchEncoding["attention_mask"]

        patch_lengths, scores = self.patcher.patch(
            tokens=tokens,
            include_next_token=False,  
            entropies=attention_mask,
        )

        num_patches = patch_lengths.shape[1]
        offset = 0
        full_patch_lengths = patch_lengths
        if self.model_type == "clm":
            max_num_patches = self.max_num_patches - 1
        else:
            max_num_patches = self.max_num_patches

        if num_patches > max_num_patches:
            if self.random_truncation:
                offset = torch.randint(0, num_patches - max_num_patches + 1, (1,), generator=self.rand_generator).item()
            else:
                offset = num_patches - max_num_patches
            patch_lengths = patch_lengths[:, offset:offset + max_num_patches]
        if self.model_type == "clm":
            if patch_lengths[0,0] > 1:
                patch_lengths[0,0] -= 1
                patch_lengths = torch.cat([torch.ones(1,1, dtype=torch.long), patch_lengths], dim=1)
        num_patches = patch_lengths.shape[1]
        if num_patches < self.max_num_patches:
            padded_patch_lengths = torch.zeros(1, self.max_num_patches, dtype=torch.long, device=patch_lengths.device)
            padded_patch_lengths[:, :num_patches] = patch_lengths
            patch_lengths2 = padded_patch_lengths
            patch_lengths = torch.cat([patch_lengths, torch.zeros(1, self.max_num_patches-num_patches, dtype=torch.long)], dim=1)
            assert torch.all(patch_lengths == patch_lengths2)
        
        assert patch_lengths.shape[1] == self.max_num_patches
        res_offset = torch.sum(full_patch_lengths[:, :offset])
        res_end = res_offset + torch.sum(patch_lengths)
        input_ids = batchEncoding["input_ids"][:, res_offset:res_end]
        attention_mask = batchEncoding["attention_mask"][:, res_offset:res_end]

        seq_length = input_ids.shape[1]
        pad_len = self._max_seq_length - seq_length

        if pad_len > 0:
            padded_input_ids = torch.full((1, self._max_seq_length), self.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
            padded_input_ids[:, :seq_length] = input_ids
            padded_attention_mask = torch.full((1, self._max_seq_length), 0, dtype=torch.long, device=attention_mask.device)
            padded_attention_mask[:, :seq_length] = attention_mask
            input_ids = torch.cat([input_ids, torch.ones(1, self._max_seq_length - seq_length, dtype=torch.long) * self.tokenizer.pad_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(1, self._max_seq_length - seq_length, dtype=torch.long)], dim=1)
            assert torch.all(input_ids == padded_input_ids)
            assert torch.all(attention_mask == padded_attention_mask)
        assert input_ids.shape[1] == self._max_seq_length

        batchEncoding["input_ids"] = input_ids
        batchEncoding["attention_mask"] = attention_mask
        batchEncoding["patch_lengths"] = patch_lengths

        ## To do: Enforce max residue cap M
        return batchEncoding
"""

class PatchifierTransform(Module):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        padding: Union[bool, str, PaddingStrategy] = "max_length",
        truncation: Union[bool, str, TruncationStrategy] = False,
        random_truncation: bool = False,
        max_seq_length: Optional[int] = None,
        max_num_patches: int = 512,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        patch_size: int = 4,
        patching_mode: str = "static",
        threshold: float = 0.0,
        threshold_add: float = 0.0,
        monotonicity: bool = False,
        seed: int = 42,
        model_type: str = "mlm",
        residue_mask_probability: float = 0.15,
        patch_mask_probability: float = 0.00,
    ):
        super().__init__()

        self.tokenizer = tokenizer


        self._padding = padding
        self._truncation = truncation
        self.random_truncation = random_truncation
        self._max_seq_length = max_seq_length
        self._return_token_type_ids = return_token_type_ids
        self._return_attention_mask = return_attention_mask
        self._return_overflowing_tokens = return_overflowing_tokens
        self._return_special_tokens_mask = return_special_tokens_mask
        self._return_offsets_mapping = return_offsets_mapping
        self._return_length = return_length
        self._verbose = verbose

        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.threshold = threshold
        self.threshold_add = threshold_add
        self.monotonicity = monotonicity
        self.max_num_patches = max_num_patches
        self.seed = seed
        self.patches_args = PatcherArgs(
            patch_size=self.patch_size,
            patching_mode=self.patching_mode,
            threshold=self.threshold,
            threshold_add=self.threshold_add,
            monotonicity=self.monotonicity,
            max_patch_length=self.max_num_patches,
        )

        self.patcher = Patcher(
            patcher_args=self.patches_args
        )

        self.rand_generator = torch.Generator().manual_seed(self.seed)
        self.model_type = model_type
        self.residue_mask_probability = residue_mask_probability
        self.patch_mask_probability = patch_mask_probability

        assert model_type == "clm" or model_type == "mlm"
        assert 0.0 <= residue_mask_probability <= 1.0   
        assert 0.0 <= patch_mask_probability <= 1.0

        # ADD SPAN MASKING!!!!
    def forward(
        self,
        text: Union[str, List[str], List[int]],
        entropies: Optional[torch.Tensor] = None,
    ) -> BatchEncoding:
        batchEncoding = self.tokenizer(
            text,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_seq_length,
            return_tensors="pt",
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=self._return_overflowing_tokens,
            return_special_tokens_mask=self._return_special_tokens_mask,
            return_offsets_mapping=self._return_offsets_mapping,
            return_length=self._return_length,
            verbose=self._verbose,
        )

        input_ids = batchEncoding["input_ids"]
        attention_mask = batchEncoding["attention_mask"]

        patch_lengths, scores = self.patcher.patch(
            tokens=input_ids[input_ids != self.tokenizer.pad_token_id].unsqueeze(0),
            include_next_token=False,  
            entropies=entropies,
        )
        num_patches = patch_lengths.shape[1]
        offset = 0
        if self.model_type == "clm":
            max_num_patches = self.max_num_patches - 1
        else:
            max_num_patches = self.max_num_patches

        if num_patches > max_num_patches:
            full_patch_lengths = patch_lengths
            if self.random_truncation:
                offset = torch.randint(0, num_patches - max_num_patches + 1, (1,), generator=self.rand_generator).item()
            else:
                offset = num_patches - max_num_patches
            patch_lengths = patch_lengths[:, offset:offset + max_num_patches]

            res_offset = torch.sum(full_patch_lengths[:, :offset])
            res_end = res_offset + torch.sum(patch_lengths)
            input_ids = input_ids[:, res_offset:res_end]
            attention_mask = attention_mask[:, res_offset:res_end]

            seq_length = input_ids.shape[1]
            input_ids_padded = torch.full((1, self._max_seq_length), self.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
            input_ids_padded[:, :seq_length] = input_ids
            input_ids = input_ids_padded
            
            #attention_mask = torch.full((1, self._max_seq_length), False, dtype=torch.long, device=attention_mask.device)
            #attention_mask[:, :seq_length] = True

        if self.model_type == "clm":
            if patch_lengths[0,0] > 1:
                patch_lengths[0,0] -= 1
                patch_lengths = torch.cat([torch.ones(1,1, dtype=torch.long), patch_lengths], dim=1)

        seq_length = input_ids[input_ids != self.tokenizer.pad_token_id].shape[0]
        num_patches = patch_lengths.shape[1]
        
        if num_patches < self.max_num_patches:
            padded_patch_lengths = torch.zeros(1, self.max_num_patches, dtype=torch.long, device=patch_lengths.device)
            padded_patch_lengths[:, :num_patches] = patch_lengths
            patch_lengths = padded_patch_lengths


        if self.model_type == "mlm":
            pad_mask = input_ids == self.tokenizer.pad_token_id
            cls_mask = input_ids == self.tokenizer.cls_token_id
            eos_mask = input_ids == self.tokenizer.eos_token_id

            pad_patch_mask = patch_lengths == 0

            masked_ids = torch.full(size = input_ids.shape, fill_value = False, device = input_ids.device)
            if self.patch_mask_probability > 0.0:
                probability_matrix = torch.full(patch_lengths.shape, self.patch_mask_probability)
                probability_matrix.masked_fill_(pad_patch_mask, 0.0)
                ## FIX PADDED PATCHES!!!
                masked_patch_ids = torch.bernoulli(probability_matrix).bool()
                end = patch_lengths.cumsum(dim=1)[masked_patch_ids]
                start = end - patch_lengths[masked_patch_ids]
                masked_ids[:, start:end] = True

                masked_ids = masked_ids & ~ pad_mask & ~ cls_mask & ~ eos_mask

            elif self.residue_mask_probability > 0.0:
                probability_matrix = torch.full(input_ids.shape, self.residue_mask_probability)
                probability_matrix.masked_fill_(pad_mask | cls_mask | eos_mask | masked_ids, 0.0)
                masked_ids = torch.bernoulli(probability_matrix).bool()

            labels = input_ids.clone()
            labels[~masked_ids] = -100

            replaced_ids = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_ids
            input_ids[replaced_ids] = self.tokenizer.mask_token_id

            random_ids = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_ids & ~ replaced_ids
            input_ids[random_ids] = torch.randint(4, self.tokenizer.vocab_size, input_ids[random_ids].shape, 
                                                  device=input_ids.device,
                                                  generator=self.rand_generator)
            assert input_ids.shape[1] == self._max_seq_length

        elif self.model_type == "clm":
            labels = torch.full((1, self._max_seq_length-1), -100, dtype=torch.long, device=input_ids.device)
            labels[:, :seq_length-1] = input_ids[:,1:seq_length]
            input_ids[:, seq_length-1:] = self.tokenizer.pad_token_id
            input_ids = input_ids[:, :self._max_seq_length-1]
            assert input_ids.shape[1] == self._max_seq_length-1
            #patch_lengths[:, num_patches-1] -= 1
            #assert torch.all(patch_lengths[:, num_patches-1] >= 0)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
        attention_mask = input_ids != self.tokenizer.pad_token_id
        attention_mask_patch = patch_lengths != 0

        assert patch_lengths.shape[1] == self.max_num_patches
        assert attention_mask_patch.shape[1] == self.max_num_patches

        batchEncoding["input_ids"] = input_ids
        batchEncoding["attention_mask"] = attention_mask.bool()
        #batchEncoding["attention_mask"] = torch.where(attention_mask.bool(), 0.0, float("-inf"))
        batchEncoding["patch_lengths"] = patch_lengths
        batchEncoding["attention_mask_patch"] = attention_mask_patch.bool()
        #batchEncoding["attention_mask_patch"] = torch.where(attention_mask_patch.bool(), 0.0, float("-inf"))
        batchEncoding["labels"] = labels

        ## To do: Enforce max residue cap M
        return batchEncoding