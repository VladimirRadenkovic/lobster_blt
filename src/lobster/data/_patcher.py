import math
import time
from enum import Enum
from collections import defaultdict

import torch
from torch.nn import functional as F

import json


def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy

def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask

def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask

def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()

def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths


def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]
    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[
        1
    ]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        # patch_start_mask[1:] |= tokens[:-1] < OFFSET
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids

def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))

def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids

   

class PatchingModeEnum(str, Enum):
    """
    Enum for the different types of patches that can be used.
    """
    ENTROPY = "entropy"
    STATIC = "static"
    RESIDUE = "residue"

class PatcherArgs:
    """
    Args for the Patcher class.
    """
    def __init__(self,
                 patching_mode: str | PatchingModeEnum = PatchingModeEnum.ENTROPY,
                 patching_device: str = "cuda",
                 realtime_patching: bool = False,
                 entropy_model_checkpoint_dir: str = None,
                 patch_size: int = 4.5,
                 threshold: float = 1.335442066192627,
                 threshold_add: float = None,
                 max_patch_length: int = None,
                 patching_batch_size: int = 1,
                 device: str = "cuda",
                 monotonicity: bool = False,
                 log_time: bool = False,
                ):
        if isinstance(patching_mode, str):
            patching_mode = PatchingModeEnum(patching_mode)
        self.patching_mode = patching_mode
        self.patching_device = patching_device
        self.realtime_patching = realtime_patching
        self.entropy_model_checkpoint_dir = entropy_model_checkpoint_dir
        self.patch_size = patch_size
        self.threshold = threshold
        self.threshold_add = threshold_add
        self.max_patch_length = max_patch_length
        self.patching_batch_size = patching_batch_size
        self.device = device
        self.monotonicity = monotonicity
        self.log_time = log_time


def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst


class Patcher:
    """
    A class for patching sequences.
    """
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.realtime_patching = patcher_args.realtime_patching
        if self.realtime_patching:
            assert (
                patcher_args.entropy_model_checkpoint_dir is not None
            ), "Cannot require realtime patching without an entropy model checkpoint"
            # TODO: Load the entropy model
            #self.entropy_model = LobsterEntropyPCLM.from_pretrained(patcher_args.entropy_model_checkpoint_dir)
            #self.entropy_model.to(patcher_args.patching_device)
            #self.entropy_model.eval()
        else:
            self.entropy_model = None
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time
        if self.log_time:
            self.log = defaultdict(float)
        
    def patch(
            self,
            tokens: torch.Tensor,
            include_next_token: bool = False,
            preds: torch.Tensor = None,
            entropies: torch.Tensor = None,
            threshold: float = None,
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        Patching with the following modes:
        1. patching_mode = None: static patch size
        2. patching_mode = "entropy":
            calculate entropy of each token, allocate patches so that the total
            number of patches is the same as static patching but choose to begin
            patches on tokens where the model is most uncertain (highest entropy).

            When threshold is provided, it uses the threshold to decide when to
            start a new patch.
        3. patching_mode = "space":
            use space like tokens to define the patches.
        4. patching_mode = "bpe":
            use bpe delim tokens to define the patches.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None
        # STATIC
        if self.patching_mode == PatchingModeEnum.STATIC:
            patch_lengths = torch.zeros(
                (bs, math.ceil(seq_len_next_tok / self.patch_size)),
                dtype=tokens.dtype,
                device=tokens.device,
            ).fill_(self.patch_size)
            if seq_len_next_tok % self.patch_size != 0:
                patch_lengths[:, -1] = seq_len_next_tok % self.patch_size
        elif self.patching_mode == PatchingModeEnum.RESIDUE:
            patch_lengths = torch.ones(
                (bs, seq_len_next_tok), dtype=tokens.dtype, device=tokens.device
            )
        # ENTROPY
        elif self.patching_mode == PatchingModeEnum.ENTROPY:
            if self.log_time:
                s = time.time()
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                start_entropies = time.time()
                #scores, _ = calculate_entropies(
                 #   tokens,
                  #  self.entropy_model,
                   # self.patching_batch_size,
                   # self.device,
                #)
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - s
                s = time.time()
            patch_start_ids = find_entropy_patch_start_ids(
                entropies = scores,
                threshold = threshold if threshold is not None else self.threshold,
                threshold_add=self.threshold_add,
                monotonicity = self.monotonicity,
                include_next_token = include_next_token,
                patch_size=self.patch_size,
            )
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - s
                s = time.time()
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - s
                s = time.time()
            
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        assert not check_non_zero_after_zero(patch_lengths)
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, : patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        assert (
            torch.sum(patch_lengths)
            == tokens.numel() + include_next_token * tokens.shape[0]
        ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        return patch_lengths, scores

"""   
json_file = "src/lobster/data/entropy_figure.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Parse inner dataframe_json
dataframe = json.loads(data["dataframe_json"])

entropies = list(dataframe["entropies"].values())
tokens = list(dataframe["tokens"].values())
token_ids = list(dataframe["token_ids"].values())
threshold = data["threshold"]

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(entropies))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, entropies, color='skyblue', edgecolor='black')

# Add token labels below each bar
ax.set_xticks(x)
ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)

# Draw the threshold line
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1)
ax.text(len(entropies) - 0.5, threshold + 0.02, f'Threshold = {threshold:.2f}', color='red', fontsize=10)

# Set axis labels and title
ax.set_ylabel('Entropy')
ax.set_title('Entropy per Token with Threshold Line')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()
plt.savefig('entropy_figure.png')

# Convert entropies to a PyTorch tensor
entropies_tensor = torch.tensor(entropies).unsqueeze(0).repeat(3,1)
tokens_tensor = torch.tensor([token_ids], dtype=torch.long).repeat(3,1)
args = PatcherArgs(
    patching_mode="entropy",
    threshold=threshold,
    monotonicity=False,  # Change to True to test monotonic version
    log_time=True,       # Enable timing logs
    patching_device="cpu",  # Use CPU for debugging; switch to "cuda" if needed
)


# Initialize Patcher
patcher = Patcher(args)

# Run patch
patch_lengths, scores = patcher.patch(
    tokens=tokens_tensor,
    include_next_token=True,  
    entropies=entropies_tensor,
)

patch_lengths2, scores2 = patcher.patch(
    tokens=tokens_tensor,
    include_next_token=False,
    entropies=entropies_tensor,
)

def patch_ids_from_lengths(patch_lengths, seq_len):
    bs, num_patches = patch_lengths.shape
    # Create a tensor of cumulative sums of the patch lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"{torch.max(patch_ids)} > {patch_lengths.shape[-1]} or {torch.min(patch_ids)} < 0"
    return patch_ids


enc_patch_ids = patch_ids_from_lengths(
        patch_lengths= patch_lengths, seq_len=81
    )

dec_patch_ids = decoder_patch_ids_from_lengths(
        patch_lengths= patch_lengths, nb_boe=0, seq_len=81
    )

enc_patch_ids2 = patch_ids_from_lengths(
        patch_lengths= patch_lengths2, seq_len=81
    )

dec_patch_ids2 = decoder_patch_ids_from_lengths(
        patch_lengths= patch_lengths2, nb_boe=0, seq_len=81
    )
print('aa')
def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):

    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask

def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]

cross_attn_mask_enc = cross_attn_mask(
                enc_patch_ids,
                patch_lengths,
                81,
                patches_as_queries=True,
                cross_attn_k=1,
                window=None,
                block_mask=False,
            )

cross_attn_mask_dec = cross_attn_mask(
                dec_patch_ids,
                patch_lengths,
                81,
                patches_as_queries=False,
                cross_attn_k=1,
                window=None,
                block_mask=False,
            )


# Print results
print("Patch lengths:\n", patch_lengths)
print("Log time breakdown (in seconds):\n", dict(patcher.log))


patch_start_ids1 = find_entropy_patch_start_ids(entropies = entropies_tensor, threshold = threshold, monotonicity = False, include_next_token = True)
patch_start_ids2 = find_entropy_patch_start_ids(entropies = entropies_tensor, threshold = threshold, monotonicity = False, include_next_token = False)
patch_start_ids3 = find_entropy_patch_start_ids(entropies = entropies_tensor, threshold = threshold, monotonicity = True, include_next_token = True)
patch_start_ids4 = find_entropy_patch_start_ids(entropies = entropies_tensor, threshold = threshold, monotonicity = True, include_next_token = False)

print('done')


"""