import torch
import pytest
from xformers.ops import fmha
from src.lobster.model.lm_base.base_transformer import Attention, FeedForward, precompute_freqs_cis, RotaryEmbedding, TransformerBlock, BaseTransformer


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
        attn_impl: str | None = "sdpa",
        attn_bias_type: str | None = None,
        sliding_window: int | None = 128,
        model_type: str | None = "mlm",
) -> torch.Tensor:
    
    if attn_bias_type is None and attn_impl == "xformers":
        mask = torch.where(attn_mask, 0.0, float("-inf"), device=device)
        # Incositency: no query padding!!!
        return fmha.attn_bias.LowerTriangularMaskWithTensorBias(mask.squeeze(1), device=device)
    
    if attn_bias_type is None:
        if model_type == "mlm":
            mask = attn_mask
        elif model_type == "clm":
            if attn_impl == "sdpa":
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
    return torch.where(mask, 0.0, float("-inf"), device=device)


def get_device():
    """Helper function to get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def test_attention_forward_and_backward():
    device = get_device()
    print(f"Running test on device: {device}")


def test_attention_forward_and_backward():
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 128
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0
    device = get_device()

    rope_embeddings = RotaryEmbedding(
            theta=rope_theta,
            head_dim= head_dim,
            max_seqlen=32,
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
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE
    freq_cis = rope_embeddings(seqlen=32).to(device)
    
    # Forward pass
    output = attention(x, freq_cis)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_feedforward_forward_and_backward():
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    hidden_dim = 1024
    multiple_of = 256
    ffn_dim_multiplier = None
    device = get_device()

    # Initialize feedforward module
    feedforward = FeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier
    ).to(device)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Forward pass
    output = feedforward(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_attention_with_mask():
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0
    device = get_device()

    # Initialize attention module
    attention = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta
    ).to(device)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE
    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=rope_theta).to(device)
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = ~mask  # Convert to attention mask format
    mask = mask.to(device)
    
    # Forward pass with mask
    output = attention(x, freqs_cis, mask=mask)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_transformer_block():
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0
    device = get_device()

    # Create a simple config-like object
    class Config:
        def __init__(self):
            self.dim = dim
            self.head_dim = head_dim
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.rope_theta = rope_theta
            self.multiple_of = 256
            self.ffn_dim_multiplier = None
            self.norm_eps = 1e-6

    config = Config()
    
    # Initialize transformer block
    block = TransformerBlock(config).to(device)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE
    rope_embeddings = RotaryEmbedding(
        theta=rope_theta,
        head_dim=head_dim,
        max_seqlen=seq_len,
    )
    freq_cis = rope_embeddings(seqlen=seq_len).to(device)
    
    # Forward pass
    output = block(x, freq_cis)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Test weight initialization
    block.init_weights()
    # Check that weights are initialized (not all zeros)
    assert not torch.allclose(block.attention.wq.weight, torch.zeros_like(block.attention.wq.weight))
    assert not torch.allclose(block.feed_forward.w1.weight, torch.zeros_like(block.feed_forward.w1.weight))

def test_base_transformer():
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0
    n_layers = 4
    device = get_device()

    # Create a simple config-like object
    class Config:
        def __init__(self):
            self.dim = dim
            self.head_dim = head_dim
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.rope_theta = rope_theta
            self.multiple_of = 256
            self.ffn_dim_multiplier = None
            self.norm_eps = 1e-6
            self.n_layers = n_layers
            self.max_seqlen = seq_len
            self.init_base_std = 0.02
            self.init_std_factor = 1.0
            self.attn_impl = "sdpa"
            self.attn_bias_type = "causal"
            self.rope_use_fp32_in_outer_product = False

    config = Config()
    
    # Initialize base transformer
    transformer = BaseTransformer(config).to(device)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Forward pass
    output = transformer(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)
    
    # Create dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Test weight initialization
    transformer.init_weights()
    # Check that weights are initialized (not all zeros)
    first_layer = transformer.layers[0]
    assert not torch.allclose(first_layer.attention.wq.weight, torch.zeros_like(first_layer.attention.wq.weight))
    assert not torch.allclose(first_layer.feed_forward.w1.weight, torch.zeros_like(first_layer.feed_forward.w1.weight))

if __name__ == "__main__":

    test_attention_forward_and_backward()
    test_feedforward_forward_and_backward()
    test_attention_with_mask()
    test_transformer_block()
    test_base_transformer()
    print("All tests passed!")