import torch
import pytest
from src.lobster.model.lm_base.base_transformer import Attention, FeedForward, precompute_freqs_cis

def get_device():
    """Helper function to get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def test_attention_forward_and_backward():
    device = get_device()
    print(f"Running test on device: {device}")

    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0

    # Initialize attention module and move to device
    attention = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta
    ).to(device)

    # Create input tensor on device
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE on device
    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=rope_theta).to(device)
    
    # Forward pass
    output = attention(x, freqs_cis)
    
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
    device = get_device()
    print(f"Running test on device: {device}")

    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    hidden_dim = 1024
    multiple_of = 256
    ffn_dim_multiplier = None

    # Initialize feedforward module and move to device
    feedforward = FeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier
    ).to(device)

    # Create input tensor on device
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
    device = get_device()
    print(f"Running test on device: {device}")

    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0

    # Initialize attention module and move to device
    attention = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta
    ).to(device)

    # Create input tensor on device
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE on device
    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=rope_theta).to(device)
    
    # Create causal mask on device
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask = ~mask  # Convert to attention mask format
    
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

def test_attention_large_batch():
    """Test attention with a larger batch size to stress test GPU memory."""
    device = get_device()
    print(f"Running large batch test on device: {device}")

    # Test parameters with larger batch size
    batch_size = 32
    seq_len = 128
    dim = 512
    head_dim = 64
    n_heads = 8
    n_kv_heads = 4
    rope_theta = 10000.0

    # Initialize attention module and move to device
    attention = Attention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta
    ).to(device)

    # Create input tensor on device
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True, device=device)
    
    # Precompute frequency cis for RoPE on device
    freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=rope_theta).to(device)
    
    # Forward pass
    output = attention(x, freqs_cis)
    
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

if __name__ == "__main__":
    # Run all tests
    test_attention_forward_and_backward()
    test_feedforward_forward_and_backward()
    test_attention_with_mask()
    test_attention_large_batch()
    print("All tests passed!") 