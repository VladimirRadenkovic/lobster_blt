"""Adapted from https://github.com/huggingface/transformers/tree/v4.23.1/src/transformers/models"""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
flex_attention_comp = torch.compile(flex_attention)
from xformers.ops import AttentionBias

from lobster.model.lm_base.base_transformer import TransformerBlock, BaseTransformer, RotaryEmbedding


from transformers.utils import logging

from lobster.transforms._patcher import PatcherArgs, Patcher
from ._utils import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseTransformerOutput
)
from torch.nn.attention.flex_attention import create_block_mask

from ._lm_base import LMBaseEncoder, LMBaseEmbeddings, LMBaseLayer
logger = logging.get_logger(__name__)

def fill_tokens(tokens, patch_size, fill_id):
    batch_size, seq_len = tokens.shape
    if seq_len % patch_size == 0:
        return tokens
    else:
        remaining = patch_size - seq_len % patch_size
        final_padding = tokens.new(batch_size, remaining).fill_(fill_id)
        return torch.cat((tokens, final_padding), dim=1)
    
def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: torch.Tensor,
    patch_size: int,
    boe_id: int,
):
    """
        This function returns X_et, X_gt and X_dt, the encoder, global, and decoder
    tokens respectively.

    Consider the input and target sequences:
    X=[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13]
    Y=[4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13,14]
    with patch_size=4

    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global

    Current without boe:
    X_et = [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]]
    X_g =  [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]] # remove last glob patch
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]         [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    --> lag fix:
    X_et = [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11] [12,13,pad,pad]]
    X_g =  [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11]]
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]    	  [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    Dynamic (current):
    X = [3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    Y = [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    entropy patching:
    input: 7, bos, 9, 10
    pred (high entropy): eos, 8, 10, eos

    X_et = [[boe,3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    X_g =  [[boe],      [3,4,5,6], [7,eos],[bos,8],[9],     [10,eos]]
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8],[9],    [10,eos],[bos]]
    Y =    [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    --> lag fix no boe (force single byte first patch):
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4,5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    input: 4, 7, bos, 9, 10
    pred (high entropy): 5, eos, 8, 10, eos

    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    Handle the last byte properly.
    patch_lengths = [1, 1,         3,      2,         2      1           2               2         1]
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # do not remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11]       [12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,        13]]


    bpe delim
    X_et = [[3,4,5,6,7,<d>,eos,bos,<d>,8,9,<d>,10,<d>,eos,bos,11,12]
    X_g =  [[3],          [4,5,6,7,<d>],     [eos,bos,<d>], ..
    X_dt = [[3,4,5,6,7],  [<d>,eos,bos],     [<d>,bos,8], ..
    Y =    [4,5,6,7,<d>,    eos,bos,<d>       8,9,<d>, ..


    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global
    """
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)
    # global_tokens = tokens.new(batch_size, ((seq_len-1) // patch_size)+1).fill_(boe_id)

    # create global tokens, contains boe tokens and eos
    # padded_local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)
    # patches = padded_local_encoder_tokens.view(batch_size, -1, patch_size)
    # global_tokens = (patches.eq(eos_id).any(dim=2).int() * eos_id)[:, 1:]
    # global_tokens += global_tokens.eq(0).int() * boe_id
    # TODO: fix this when we want to use block causal in the global.

    if enforce_patch_size_multiple and local_encoder_tokens.shape[-1] % patch_size != 0:
        local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)

    return local_encoder_tokens, None, local_decoder_tokens

def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


def patch_reduce(hidden_states, max_num_patches, reduction, patch_ids):
    """
    Reduce variable length patches to single embedding per patch
    Note: this works with variable number of patches for different sequences in the batch
    It handles variable length patches by assuming that patch_lengths will be 0 for any
    extra patches on the *right*. Since there can be a variable number of patches
    this function also return the number of patches for each sequence in the batch.
    Any embeddings on the right that are not allocated to a patch
    (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
    will be sent to a dummy patch, which is trimmed before returning.
    """

    batch_size, seq_len, emb_dim = hidden_states.shape
    patch_ids = patch_ids.clone()
    valid_mask = (patch_ids != -1)
    patch_ids[~valid_mask] = max_num_patches
    hidden_states = hidden_states * valid_mask.unsqueeze(-1)

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])   

    reduced_embs = torch.zeros(
        (batch_size, max_num_patches + 1, emb_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    reduced_embs = reduced_embs.scatter_reduce(
        src=hidden_states,
        dim=1,
        index=patch_ids,
        reduce=reduction,
        include_self=False,
    )
    reduced_embs = reduced_embs[:, :max_num_patches, :]

    return reduced_embs

def concat_downsample(hidden_states, patch_lengths, patch_size):
    # The assumption in this function is that seq_len = patch_size * num_patches.
    batch_size, seq_len, emb_dim = hidden_states.shape
    patch_end_ids = torch.cumsum(patch_lengths, dim=1)
    patch_ids = patch_end_ids.unsqueeze(-1) - torch.arange(patch_size, 0, -1).to(
        patch_end_ids.device
    )
    # Is clamp ok here?
    patch_ids = patch_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, hidden_states.shape[-1])
    patch_ids = patch_ids.view(batch_size, -1, hidden_states.shape[-1])
    # after gather h.shape = [batch_size, seq_len, dim]
    hidden_states = torch.gather(hidden_states, 1, patch_ids)
    hidden_states = hidden_states.reshape(batch_size, patch_lengths.shape[1], patch_size * hidden_states.size(-1))
    return hidden_states

def pooling_downsample(hidden_states, max_num_patches, pooling_mode, patch_ids):
    cat = []
    if "avg" in pooling_mode or "mean" in pooling_mode:
        cat.append(patch_reduce(hidden_states, max_num_patches, "mean", patch_ids))
    if "min" in pooling_mode:
        cat.append(patch_reduce(hidden_states, max_num_patches, "amin", patch_ids))
    if "max" in pooling_mode:
        cat.append(patch_reduce(hidden_states, max_num_patches, "amax", patch_ids))
    assert len(cat) > 0
    hidden_states = torch.cat(cat, dim=-1)
    return hidden_states

    
def downsample(
    hidden_states,
    num_patches,
    patch_length= None,
    patch_ids=None,
    downsampling_by_pooling=None,
    patch_size=4,
):
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        max_num_patches = num_patches
        assert patch_ids is not None
        hidden_states = pooling_downsample(hidden_states, max_num_patches, downsampling_by_pooling, patch_ids)
    else: 
        assert patch_length is not None
        hidden_states = concat_downsample(hidden_states, patch_length, patch_size)
    return hidden_states

def patch_ids_from_lengths(patch_lengths, attn_mask):
    bs, seq_lens = attn_mask.squeeze(1).shape
    # Create a tensor of cumulative sums of the patch lengths
    len_sum = patch_lengths.cumsum(dim=-1)
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            len_sum,
        ],
        dim=-1,
    )
    seq_lens = torch.arange(seq_lens, device=cum_d.device).repeat(bs,1)
    seq_lens.masked_fill_(attn_mask.squeeze(1) != 1, -1)
    patch_ids = (cum_d.unsqueeze(-1) <= seq_lens.unsqueeze(1)).sum(
        dim=-2
    ) - 1
    assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"
    return patch_ids

def decoder_patch_ids_from_lengths(patch_lengths, attn_mask):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length  == 1
    ), f"First patch (patch length: {first_patch_length}) should be of length 1)"
    decoder_patch_lengths = patch_lengths[:, 1:]
    
    assert (
        decoder_patch_lengths.sum() +  patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, attn_mask=attn_mask
    )

    return decoder_patch_ids

def create_patch_mask_from_ids(
    patch_ids, num_patches, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
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
    
    mask = q_ids == kv_ids

    return mask


def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        if block_mask:

            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                _compile=True,
            )
            return block_mask
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]

def sliding_window_mask(bs, seq_len, left_window, right_window, device=None):
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # shape: (seq_len, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # shape: (1, seq_len)
    mask = (j >= (i - left_window)) & (j <= (i + right_window))
    return mask.unsqueeze(0).expand(bs, -1, -1) 


def create_mask(
        bs: int,
        seq_len: int,
        device: torch.device,
        attn_mask: torch.Tensor,
        attn_bias_type: str | None = None,
        sliding_window: int | None = 128,
        model_type: str | None = "mlm",
) -> torch.Tensor:  
    if attn_bias_type is None:
        if model_type == "mlm":
            mask = attn_mask
        elif model_type == "clm":
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
            mask = mask.unsqueeze(0).expand(bs, -1, -1) 
    elif attn_bias_type == "sliding_window":
        if model_type == "mlm":
            mask = sliding_window_mask(bs, seq_len, sliding_window, sliding_window, device=device)
        elif model_type == "clm":
            mask = sliding_window_mask(bs, seq_len, sliding_window, 0, device=device)
    else:
        raise ValueError(f"Invalid attn_bias_type: {attn_bias_type}")
        
    mask = mask & attn_mask.expand(-1, seq_len, -1) & attn_mask.squeeze(1).unsqueeze(-1) # key, query padding
    return torch.where(mask.unsqueeze(1), torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))

class GlobalTransformer(BaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = config.dropout
        self.dim_token_emb = config.dim_token_emb

        self.token_embedding_projection = None

        if config.dim_token_emb is not None and config.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                config.dim_token_emb,
                config.dim,
                bias=False,
            )

    def forward(
        self,
        embeds: Optional[torch.Tensor] = None,
        attn_mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
    ):
        bs, seqlen, _ = embeds.shape

        h = embeds

        mask = create_mask(
            bs, 
            seqlen, 
            h.device, 
            attn_mask=attn_mask, 
            attn_bias_type=self.attn_bias_type, 
            model_type=self.model_type)

        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        h = super().forward(h, mask=mask, attn_impl=self.attn_impl)
        return h

    def init_weights(self):
        super().init_weights()



        

class LatentTransfromer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.dim_token_emb = config.dim_token_emb
        if config.dim_token_emb is not None and config.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                config.dim_token_emb,
                config.dim,
                bias=False,
            )
        self.encoder = LMBaseEncoder(config)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
      
        input_shape = embeddings.size()[:-1]
 
        batch_size, seq_length, _ = embeddings.shape
        device = embeddings.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if self.token_embedding_projection is not None and embeddings.shape[-1] != self.dim:
            hidden_states = self.token_embedding_projection(embeddings)
        else:
            hidden_states = embeddings
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        return (sequence_output) + encoder_outputs[1:]




        ## BLOCK
"""
class LMLatentBaseModel(LMBasePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = False

    # Copied from transformers.models.LMBase.modeling_LMBase.LMBaseModel.__init__ with LMBase->LMBase
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.dim_token_emb = config.dim_token_emb

        if config.dim_token_emb is not None and config.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                config.dim_token_emb,
                config.dim,
                bias=False,
            )
        ## BLOCK-CAUSAL-MASK???
        self.encoder = LMBaseEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LMBaseEncoder):
            module.gradient_checkpointing = value
f
    def _prune_heads(self, heads_to_prune):
       
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
      
        input_shape = inputs_embeds.size()[:-1]
 
        batch_size, seq_length = input_shape
        device = inputs_embeds.device



        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if self.token_embedding_projection is not None and inputs_embeds.shape[-1] != self.dim:
            h = self.token_embedding_projection(inputs_embeds)
        else:
            h = inputs_embeds
        h = F.dropout(h, p=self.dropout, training=self.training)
        encoder_outputs = self.encoder(
            h,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseTransformerOutput(

            logits=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

"""

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = nn.RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x_norm = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x_norm)
        xk= self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        assert mask is None or isinstance(mask, BlockMask)

        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        output = self.wo(output.reshape(output_shape))

        return x + output

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.wq.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wk.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wv.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()    

"""
class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_kv_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.num_kv_attention_heads = num_kv_attention_heads
        self.num_heads_per_group = self.num_attention_heads // self.num_kv_attention_heads

        self.cross_attention_norm_query = nn.LayerNorm(hidden_size) # TODO: RMSNorm      
        self.cross_attention_norm_key_value = nn.LayerNorm(hidden_size) # TODO: RMSNorm  

        self.attention_head_size = int(hidden_size / num_attention_heads)  # must be divisible
        self.query_all_head_size = self.num_attention_heads * self.attention_head_size
        self.key_value_all_head_size = self.num_kv_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.query_all_head_size, bias= False) # BIAS FALSE!!!
        self.key = nn.Linear(hidden_size, self.key_value_all_head_size, bias=False) # BIAS FALSE!!!
        self.value = nn.Linear(hidden_size, self.key_value_all_head_size, bias=False) # BIAS FALSE!!!
        self.output = nn.Linear(hidden_size, hidden_size, bias=False) # BIAS FALSE!!!

        self.attention_probs_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_kv: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        _, seq_length_kv, _ = hidden_states_kv.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)

        hidden_states = self.cross_attention_norm_query(hidden_states)
        hidden_states_kv = self.cross_attention_norm_key_value(hidden_states_kv)

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states_kv)
        value_layer = self.value(hidden_states_kv)

        

        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_length_kv, self.num_kv_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, seq_length_kv, self.num_kv_attention_heads, self.attention_head_size)
    
        query_layer = query_layer * self.attention_head_size**-0.5


        
        key_layer = repeat_kv(key_layer, self.num_heads_per_group, dim=2)
        value_layer = repeat_kv(value_layer, self.num_heads_per_group, dim=2)

        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)   

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(attention_probs) # SHOULD BE DROPOUT 0.0


        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.query_all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = self.output(context_layer)
        outputs = self.hidden_dropout(outputs)
        outputs = hidden_states + outputs

        outputs = (outputs, attention_probs) if output_attentions else (outputs,) ##LAYER NORM in next layer

        return outputs
        
 """       
        

class LocalModelBase(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_type = config.model_type
        self.dim = config.dim
        self.dim_patch_emb = config.dim_patch_emb
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.patch_size = config.patch_size

        self.attn_impl = config.attn_impl
        self.attn_bias_type = config.attn_bias_type
        self.sliding_window = config.sliding_window
        self.init_base_std = config.init_base_std
        self.cross_attn = config.cross_attn
        self.cross_attn_k = config.cross_attn_k 

        self.pad_token_id = config.pad_token_id

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.patch_embedding_projection = nn.Linear(
            in_features=self.dim_patch_emb,
            out_features=self.dim * self.cross_attn_k,
            bias=False, 
        )

        self.rope = RotaryEmbedding(
                theta=config.rope_theta,
                head_dim=config.head_dim or config.dim // config.n_heads,
                max_seqlen=config.max_seqlen,
                rope_use_fp32_in_outer_product=config.rope_use_fp32_in_outer_product,
            )


    def init_weights(self, init_std=None):
        self.rope.reset_parameters()
        if hasattr(self, "norm"):
            self.norm.reset_parameters()

        init_std = init_std or (self.dim ** (-0.5))

        if hasattr(self, "tok_embeddings"):
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        for depth, layer in enumerate(self.layers):
            depth_factor = (2 * (depth + 1)) ** 0.5
            layer.init_weights(None, depth_factor)

        if hasattr(self, "output"):
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.patch_embedding_projection is not None:
            patch_emb_std = self.dim_patch_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.patch_embedding_projection.weight,
                mean=0.0,
                std=patch_emb_std,
                a=-3 * patch_emb_std,
                b=3 * patch_emb_std,
            )
        if self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                depth_factor = (2 * (depth + 1)) ** 0.5
                layer.init_weights(None, depth_factor)





class LocalEncoder(LocalModelBase):
    def __init__(self, config):
        super().__init__(config)

        self.downsampling_by_pooling = config.downsampling_by_pooling
        self.expects_hash_embeddings = config.encoder_hash_byte_group_size is not None
        self.cross_attn = config.cross_attn
        self.cross_attn_all_layers = config.cross_attn_all_layers
        self.cross_attn_init_by_pooling = config.cross_attn_init_by_pooling
        self.cross_attn_nheads = config.cross_attn_nheads

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim, padding_idx=self.pad_token_id)

        if self.cross_attn:
            self.cross_attn_layers = nn.ModuleList()
            layers_to_add = config.n_layers if self.cross_attn_all_layers else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim = self.dim,
                        head_dim = self.dim // self.cross_attn_nheads,
                        n_heads = self.cross_attn_nheads,
                        n_kv_heads = self.cross_attn_nheads,
                        norm_eps = config.norm_eps
                    )
                )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        The description in the paper should be updated to reflect what is in the code. 
        We are not using the embeddings to initialize the encoder cross attention. 
        We are using the hidden states after applying the first encoder layer or 
        the full encoder depending on the configuration.
        """
        bs, seqlen = tokens.shape
        mask = create_mask(
            bs = bs,
            seq_len = seqlen,
            device = tokens.device,
            attn_mask = attn_mask,
            attn_bias_type = self.attn_bias_type,
            sliding_window = self.sliding_window,
            model_type = self.model_type
        )
        
        if embeds is None:
            h = self.tok_embeddings(tokens)
        else:
            h = embeds

        h = F.dropout(h, p=self.dropout, training=self.training)    

        freqs_cis = self.rope(seqlen = seqlen)
        patch_embeds = None
        
        for i, layer in enumerate(self.layers):
            h = layer(h, mask = mask, freq_cis = freqs_cis, attn_impl = self.attn_impl)

            if self.cross_attn and (i == len(self.layers) - 1 or self.cross_attn_all_layers):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn else None
        return h, h_residual
    
    def apply_cross_attention(
        self, h, patch_embeds, layer_idx,  bs, num_patches, patch_ids, cross_mask
    ):
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h, 
                num_patches, 
                patch_ids = patch_ids, 
                downsampling_by_pooling = self.downsampling_by_pooling, 
                patch_size = self.patch_size
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )
        
        layer_idx = layer_idx if self.cross_attn_all_layers else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x = patch_embeds,
            kv = h,
            mask = cross_mask,
        )

        return patch_embeds + patch_embeds_cross

            
    """     
    def apply_cross_attention(
            self, 
            hidden_states, 
            patch_embeddings,
            cross_attention_mask,
            layer_idx,
            batch_size,
            num_patches,
            patch_ids):

        if self.cross_attn_init_by_pooling and patch_embeddings is None:
            patch_embeddings = downsample(
                hidden_states,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeddings)
                patch_embeds = patch_embeds.reshape(
                    batch_size, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeddings_cross = self.cross_attn_layers[layer_idx](
            hidden_states=patch_embeddings,
            hidden_states_kv=hidden_states,
            attention_mask=cross_attention_mask,
            output_attentions=False,
        )
        patch_embeddings_cross = patch_embeddings_cross[0]
        return patch_embeddings + patch_embeddings_cross
    """  
        
    
class LocalDecoder(LocalModelBase):
    def __init__(self, config):
        super().__init__(config)

        self.cross_attn = config.cross_attn
        self.cross_attn_all_layers = config.cross_attn_all_layers
        self.cross_attn_nheads = config.cross_attn_nheads

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps) #RMSNorm?

        if self.cross_attn:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = config.n_layers if self.cross_attn_all_layers else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads = self.cross_attn_nheads,
                        norm_eps=config.norm_eps
                        )
                )


        self.output = nn.Linear(
            self.dim,
            self.vocab_size,
            bias=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    
    ):
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided"
        mask = create_mask(
            bs = bs,
            seq_len = seqlen,
            device = tokens.device,
            attn_mask = attn_mask,
            attn_bias_type = self.attn_bias_type,
            sliding_window = self.sliding_window,
            model_type = self.model_type
        )

        #if mask is None:
         #   mask = torch.ones((bs, seq_len), device=tokens.device)

        h = embeds

        if self.patch_embedding_projection is not None:
            assert patch_embeds is not None, "Patch embeddings must be passed."
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        if patch_embeds is not None and not self.cross_attn:
            h = h + patch_embeds

        freqs_cis = self.rope(seqlen=seqlen)

        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            if self.cross_attn and (
                i == 0 or self.cross_attn_all_layers
            ):
                # Use cross attention to extract info from patch_embeds into h
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)

        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds


## TO DO: ADD INITIALIZATION, RMSNORM, DROPOUT, MLM, CLM MASKING, ENCODER HASHING

class ByteLatentTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        self.weight_tying = config.weight_tying
        self.patch_size = config.patch_size
        self.patching_mode = config.patching_mode
        self.downsampling_by_pooling = config.downsampling_by_pooling
        self.patching_threshold = config.patching_threshold
        self.init_base_std = config.init_base_std
        
        self.max_seq_len = config.max_seqlen

        self.cross_attn_encoder = config.encoder.cross_attn
        self.cross_attn_decoder = config.decoder.cross_attn
        self.cross_attn_k = config.cross_attn_k


        self.cross_attn_use_flex_attention = config.cross_attn_use_flex_attention


        self.local_encoder = LocalEncoder(config.encoder)
        self.global_transformer = GlobalTransformer(config.global_transformer)
        self.local_decoder = LocalDecoder(config.decoder)

        # Encoder hash configuration???
        if config.patch_in_forward:
            self.patcher = Patcher(PatcherArgs(
                patch_size= config.patch_size,
                patching_mode=config.patching_mode,
                threshold=config.patching_threshold,
                threshold_add=config.patching_threshold_add,
                monotonicity=config.monotonicity,
                max_patch_length=config.max_patch_length,
            )
            )
        
    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        attn_patch_mask: Optional[torch.Tensor] = None,
        ngram_ids: Optional[torch.Tensor] = None,
    ):
        bs, N = tokens.shape
        local_encoder_tokens = tokens
        local_decoder_tokens = tokens
        if attn_mask is None:
            attn_mask = torch.ones_like(tokens, dtype=torch.bool).unsqueeze(1)

        if patch_lengths is None:
            assert (
                getattr(self, "patcher", None) is not None
            ), "Patcher not defined and no patch_lengths passed."
            patch_lengths, tok_scores = self.patcher.patch(
                local_encoder_tokens,
                include_next_token=True, #CHECK FOR THIS
                threshold=self.patcher.threshold,
            )
            
        if attn_patch_mask is None:
            attn_patch_mask = torch.ones_like(patch_lengths, dtype=torch.bool).unsqueeze(1)

        assert torch.min(patch_lengths) >= 0
        # Generate patch IDs from patch_lengths
        patch_ids = patch_ids_from_lengths(
            patch_lengths, attn_mask
        )
        
        assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"

        cross_attn_mask_enc = None

        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                N,
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                block_mask=self.cross_attn_use_flex_attention,
            )

        local_encoder_embeds = self.local_encoder.tok_embeddings(local_encoder_tokens) #N-GRAM EMBEDDINGS???
        h_encoder, h_cross = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            attn_mask=attn_mask,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        h = h_cross.view(bs, patch_lengths.shape[1], -1)
        h = self.global_transformer(
            embeds = h,
            attn_mask=attn_patch_mask,
        )
        dec_embeds = h_encoder
        if self.model_type == "mlm":
            dec_patch_ids = patch_ids
        elif self.model_type == "clm":
            dec_patch_ids = decoder_patch_ids_from_lengths(
                patch_lengths=patch_lengths, attn_mask=attn_mask
            )
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        assert (
            torch.max(dec_patch_ids) + 1 <= h.shape[1]
        ), f"{torch.max(dec_patch_ids) + 1} > {h.shape[1]}"
        assert (
            dec_patch_ids.shape[1] == dec_embeds.shape[1]
        ), f"{dec_patch_ids.shape[1]} != {dec_embeds.shape[1]}"

        if self.cross_attn_decoder:
            cross_attn_mask_dec = cross_attn_mask(
                dec_patch_ids,
                patch_lengths,
                N,
                patches_as_queries=False,
                cross_attn_k=self.cross_attn_k,
                block_mask=self.cross_attn_use_flex_attention,
            )
        # Local decoder
        output = self.local_decoder(
            embeds=dec_embeds,
            patch_embeds=h,
            tokens=local_decoder_tokens,
            attn_mask=attn_mask,
            cross_mask=cross_attn_mask_dec,
        )
        return output

    def init_weights(self):
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()