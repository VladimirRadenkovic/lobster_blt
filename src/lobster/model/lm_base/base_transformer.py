from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import BlockMask

from xformers.ops.fmha import AttentionBias, memory_efficient_attention
from torch.nn.functional import scaled_dot_product_attention
from xformers.ops import fmha
def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_use_fp32_in_outer_product: bool = False,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if rope_use_fp32_in_outer_product:
        t = t.to(torch.float32)

    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)

def apply_rotrary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        seq_dim: int,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2

    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim).float() # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
    

# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        rope_use_fp32_in_outer_product: bool = False,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=head_dim,
                end=max_seqlen,
                theta=theta,
                rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
            ),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]
        
def _reshape_for_attn_bias(
    attn_bias: AttentionBias | None,
    *tensors: torch.Tensor,
) -> list[torch.Tensor]:
    to_transform = list(tensors)
    return to_transform



class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads


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
            freq_cis: torch.Tensor,
            tok_idx: Optional[Union[BlockMask, AttentionBias, str]] = None,
            mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
            attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B, S, D
        bsz, seq_len, dim = x.shape

        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotrary_emb(xq, xk, 1, freq_cis[0:seq_len])

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)
        
        #assert mask is None or isinstance(mask, AttentionBias)
        query_shape = xq.shape
        #xq, xk, xv = _reshape_for_attn_bias(mask, xq, xk, xv)
        if attn_impl == "xformers":
            if isinstance(mask, fmha.AttentionBias):
                output = memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            elif isinstance(mask, torch.Tensor):
                output = memory_efficient_attention(xq, xk, xv, attn_bias=mask.expand(-1,self.n_heads,-1,-1))
            else:
                raise ValueError(f"Mask type {type(mask)} not supported")
        else:
            output = scaled_dot_product_attention(query=xq.transpose(1, 2), 
                                                  key=xk.transpose(1, 2), 
                                                  value=xv.transpose(1, 2), 
                                                  attn_mask=mask).transpose(1, 2).contiguous() 
        output = output.view(query_shape)
            # This uses B S H D instead of B H S D of pytorch

        output = self.wo(output.reshape(output_shape))

        return output
        
    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5)) / factor

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, S, D

        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)

        return output
    
    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5)) / factor
        out_init_std = init_std or (self.hidden_dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()


        assert (config.head_dim is not None) or (
            config.n_heads is not None
        ), "Either head_dim or n_heads must be provided"
        
        self.head_dim = config.head_dim or config.dim // config.n_heads
        self.n_heads = config.n_heads or config.dim // config.head_dim
        self.n_kv_heads = config.n_kv_heads or self.n_heads

        assert config.n_heads % self.n_kv_heads == 0
        assert config.dim % config.n_heads == 0

        self.attention = Attention(
            dim=config.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=config.rope_theta,
        )

        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )

        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)


    def forward(
            self,
            x,
            freq_cis: Optional[torch.Tensor] = None,
            tok_idx: Optional[torch.Tensor] = None,
            mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
            attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        
        attn_out = self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        x = x + attn_out
        h_norm = self.ffn_norm(x)
        out = x + self.feed_forward(h_norm)
        
        return out
    
    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()
        

        
class BaseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.init_base_std = config.init_base_std
        self.attn_impl = config.attn_impl
        self.attn_bias_type = config.attn_bias_type
        self.model_type = config.model_type
        self.max_seqlen = config.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta = config.rope_theta,
            head_dim = config.head_dim or config.dim // config.n_heads,
            max_seqlen = config.max_seqlen,
            rope_use_fp32_in_outer_product = config.rope_use_fp32_in_outer_product,
        )
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))
        
        
    def forward(
            self, 
            h,
            tok_idx: Optional[torch.Tensor] = None,
            mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
            attn_impl: str = "sdpa",
            ):
        
        freq_cis = self.rope_embeddings(self.max_seqlen, tok_idx)
        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx, mask, attn_impl)
        return h
    
    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        for depth, layer in enumerate(self.layers):
            depth_factor = (2 * (depth + 1)) ** 0.5
            layer.init_weights(self.init_base_std, depth_factor)
        
        