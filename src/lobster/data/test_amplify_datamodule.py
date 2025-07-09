import torch
from pathlib import Path
from lobster.data import AmplifyLightningDataModule
from xformers.ops import fmha

from src.lobster.model.lm_base.base_transformer import Attention, RotaryEmbedding

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
"""
def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, seq_len):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"
    # Remove first patch from patch_ids for local decoder inputs and shift the last patch.
    # decoder_patch_lengths = patch_lengths[:, 1:].clone()
    # decoder_patch_lengths = add_to_last_nonzero_patch(decoder_patch_lengths, 1)
    decoder_patch_lengths = patch_lengths[:, 1:]
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )
    return decoder_patch_ids
"""
def patch_ids_from_lengths2(patch_lengths, seq_len):
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

def patch_ids_from_lengths(patch_lengths, tokens):
    bs, seq_lens = tokens.shape
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
    seq_lens.masked_fill_(tokens == 1, -1)
    patch_ids = (cum_d.unsqueeze(-1) <= seq_lens.unsqueeze(1)).sum(
        dim=-2
    ) - 1
    assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"
    return patch_ids

    


def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, tokens):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"
    # Remove first patch from patch_ids for local decoder inputs and shift the last patch.
    # decoder_patch_lengths = patch_lengths[:, 1:].clone()
    # decoder_patch_lengths = add_to_last_nonzero_patch(decoder_patch_lengths, 1)
    #assert torch.all(patch_lengths[:, 0] == 1), "first patch should be 1"
    #last_patch_id = (patch_lengths != 0).sum(dim=-1) - 1
    #last_seq_id = patch_lengths.sum(dim=-1) - 1
    decoder_patch_lengths = patch_lengths[:, 1:]
    
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, tokens=tokens
    )
    #decoder_patch_ids[torch.arange(bs), last_seq_id] = last_patch_id

    return decoder_patch_ids

def decoder_patch_ids_from_lengths2(patch_lengths, nb_boe, seq_len):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"
    # Remove first patch from patch_ids for local decoder inputs and shift the last patch.
    # decoder_patch_lengths = patch_lengths[:, 1:].clone()
    # decoder_patch_lengths = add_to_last_nonzero_patch(decoder_patch_lengths, 1)
    decoder_patch_lengths = patch_lengths[:, 1:]
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths2(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )
    
    return decoder_patch_ids

def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        window (int): If not None, only considers patches within a window of size window.
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
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window) & (kv_ids != -1) & (q_ids <= patch_ids.max(dim=1).values.unsqueeze(-1).unsqueeze(-1))
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
        """
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
        """


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

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])

    reduced_embs = torch.zeros(
        (batch_size, max_num_patches, emb_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    valid = patch_ids >= 0
    idx = torch.clamp(patch_ids, min=0)
    neg_inf = torch.finfo(hidden_states.dtype).min
    hidden_states = hidden_states.masked_fill(~valid, neg_inf)
    reduced_embs = reduced_embs.scatter_reduce(
        src=hidden_states,
        dim=1,
        index=idx,
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

    # ONLY SUPPORTS MAX AT THE MOMENT, ADD AVG, MIN
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
    patch_lengths= None,
    patch_ids=None,
    downsampling_by_pooling=None,
    patch_size=4,
):
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        max_num_patches = num_patches
        assert patch_ids is not None
        hidden_states = pooling_downsample(hidden_states, max_num_patches, downsampling_by_pooling, patch_ids)
    else: 
        assert patch_lengths is not None
        hidden_states = concat_downsample(hidden_states, patch_lengths, patch_size)
    return hidden_states


def sliding_window_mask(bs, seq_len, left_window, right_window, device=None):
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # shape: (seq_len, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # shape: (1, seq_len)
    mask = (j >= (i - left_window)) & (j <= (i + right_window))
    return mask.unsqueeze(0).expand(bs, -1, -1) 

"""
def create_mask(
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_impl: str | None = "sdpa",
        attn_bias_type: str | None = None,
        sliding_window: int | None = 128,
        model_type: str | None = "mlm",
) -> torch.Tensor:

    bs, seq_len = tokens.shape
    if attn_impl == "sdpa":
        if attn_bias_type is None:
            if model_type == "mlm":
               mask = attn_mask.expand(-1, seq_len, -1)
            elif model_type == "clm":
                mask = torch.tril(torch.ones((seq_len, seq_len), device=tokens.device, dtype=torch.bool)) # is this fine?
                mask = mask.unsqueeze(0).expand(bs, -1, -1) 
            # WARNING: EXTRA DIMENSION FROM BATCH TRANSFORM
        elif attn_bias_type == "sliding_window":
            if model_type == "mlm":
                mask = sliding_window_mask(bs, seq_len, sliding_window, sliding_window, device=tokens.device) & attn_mask.expand(-1, seq_len, -1)
            elif model_type == "clm":
                mask = sliding_window_mask(bs, seq_len, sliding_window, 0, device=tokens.device)
        return mask & attn_mask.squeeze(1).unsqueeze(-1)
    elif attn_impl == "xformers":
        if attn_bias_type is None:
            if model_type == "mlm":
                mask = attn_mask.squeeze(1) 
            elif model_type == "clm":
                attn_mask = torch.where(attn_mask, 0.0, float("-inf"), device=tokens.device)
                return fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_mask.squeeze(1), device=tokens.device)
        elif attn_bias_type == "sliding_window":
            if model_type == "mlm":
               mask = sliding_window_mask(bs, seq_len, sliding_window, sliding_window, device=tokens.device) & attn_mask.expand(-1, seq_len, -1) & attn_mask.squeeze(1).unsqueeze(-1)
            elif model_type == "clm":
                mask = sliding_window_mask(bs, seq_len, sliding_window, 0, device=tokens.device) & attn_mask.squeeze(1).unsqueeze(-1)
        return torch.where(mask, 0.0, float("-inf"), device=tokens.device)
    
"""
def create_mask(
        bs: int,
        seq_len: int,
        device: torch.device,
        attn_mask: torch.Tensor,
        attn_impl: str | None = "sdpa",
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

def get_device():
    """Helper function to get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_attention_forward_and_backward(tokens,attn_mask, attn_impl, attn_bias_type, model_type):
    # Test parameters
    batch_size = 4
    seq_len = 2048
    dim = 128
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0
    device = get_device()
    bs, seq_len = tokens.shape
    mask = create_mask(
        bs, seq_len, device, attn_mask, attn_impl, attn_bias_type, 128, model_type
    )

    rope_embeddings = RotaryEmbedding(
            theta=rope_theta,
            head_dim= head_dim,
            max_seqlen=2048,
        )
    # Initialize attention module
    attention = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta
    ).to(device)

    # Create input tensor
    emb = torch.nn.Embedding(num_embeddings=34, embedding_dim=dim).to(device)
    x = emb(tokens)
    #x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE
    freq_cis = rope_embeddings(seqlen=2048).to(device)
    
    # Forward pass
    output = attention(x=x, freq_cis=freq_cis, mask=mask, attn_impl=attn_impl)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    for name, param in attention.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient for {name}"


def debug_dataloader(dataloader, num_batches=5):
    """Function to iterate through the dataloader and print sample batches."""
    """
    patcher = Patcher(
        PatcherArgs(
            patch_size=8,
            patching_mode="static",
            threshold=0.0,
            threshold_add=0.0,
            monotonicity=False,
            max_patch_length=24,
        )
    )
    """
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i + 1}/{num_batches}:")
        tokens = batch["input_ids"].squeeze(1)
        patch_lengths = batch["patch_lengths"].squeeze(1)
        batch_size, seq_len = tokens.shape
        local_encoder_tokens = tokens
        local_decoder_tokens = tokens
        attn_mask = batch["attention_mask"]
        #test_attention_forward_and_backward(tokens, attn_mask, "sdpa", None, "mlm")
        #test_attention_forward_and_backward(tokens, attn_mask, "sdpa", None,"clm")
        #test_attention_forward_and_backward(tokens, attn_mask, "sdpa", "sliding_window", "mlm")
        #test_attention_forward_and_backward(tokens, attn_mask, "sdpa", "sliding_window", "clm")
        #test_attention_forward_and_backward(tokens, attn_mask, "xformers", None, "mlm")
        #test_attention_forward_and_backward(tokens, attn_mask, "xformers", None,"clm")
        #test_attention_forward_and_backward(tokens, attn_mask, "xformers", "sliding_window", "mlm")
        #test_attention_forward_and_backward(tokens, attn_mask, "xformers", "sliding_window", "clm")
        nb_boe = 0
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens
        )
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, 0, local_decoder_tokens
        )

        cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                local_encoder_tokens.shape[-1],
                patches_as_queries=True,
            )
        patch_lengths2 = patch_lengths[0][:29].unsqueeze(0)
        patch_ids2 = patch_ids_from_lengths2(
            patch_lengths2, 108
        )

        decoder_patch_ids2 = decoder_patch_ids_from_lengths2(
            patch_lengths2, 0, 108
        )
        patch_lengths2 = patch_lengths[1][:38].unsqueeze(0)

        patch_ids2 = patch_ids_from_lengths2(
            patch_lengths2, 146
        )
        decoder_patch_ids2 = decoder_patch_ids_from_lengths2(
            patch_lengths2, 0, 146
        )

        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, 0, local_decoder_tokens.shape[-1]
        )

        decoder_patch_ids2 = decoder_patch_ids_from_lengths2(
            patch_lengths2, 0, 147
        )

        assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"
        
        h = torch.randn(64, 2048, 512)

        patch_embeddings = downsample(h, num_patches = 512,
                                      patch_lengths = patch_lengths,
                                      patch_ids = patch_ids, 
                                      downsampling_by_pooling="max", 
                                      patch_size=4)


        cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                local_encoder_tokens.shape[-1],
                patches_as_queries=True,
            )
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, 0, local_decoder_tokens.shape[-1]
        )

        cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                local_decoder_tokens.shape[-1],
                patches_as_queries=False,
            )
        """
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=8,
            boe_id=33,
        )

        patch_lengths, tok_scores = patcher.patch(
                local_encoder_tokens,
                include_next_token=False,
                threshold=patcher.threshold,
            )
     
       
        #patch_lengths = torch.cat([patch_lengths, patch_lengths[:, -1:]], dim=1)
        # Generate patch IDs from patch_lengths
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )
        assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"

        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, local_decoder_tokens.shape[-1]
        )

        cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                seq_len,
                patches_as_queries=False,
                cross_attn_k=2,
                window=512,
                block_mask=False,
            )
        
        if isinstance(batch, dict):
            for key, value in batch.items():
                print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
        else:
            print(batch)
        """
        if i + 1 == num_batches:
            break

def main():
    # Initialize the data module with debugging settings
    data_module = AmplifyLightningDataModule(
        tokenizer_max_length=2048,  # Set a reasonable sequence length
        root=Path("/rds/user/vr375/hpc-work/lobster/data"),  # Change this to an appropriate dataset directory
        seed=42,
        batch_size=4,
        num_workers=0,  # Set to >0 for parallel data loading
        pin_memory=False,
        shuffle_buffer_size=10000,
    )

    print("\n==> Setting up the datasets")
    data_module.setup()

    # Debug training dataset
    print("\n==> Training Dataloader Debugging")
    train_loader = data_module.train_dataloader()
    debug_dataloader(train_loader)

    # Debug validation dataset
    print("\n==> Validation Dataloader Debugging")
    val_loader = data_module.val_dataloader()
    debug_dataloader(val_loader)


if __name__ == "__main__":
    main()
