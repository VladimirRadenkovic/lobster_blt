import torch
import xformers.ops as xops # Import xformers to check its availability

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}") # Assuming device 0

    # Test a simple tensor operation on GPU
    x = torch.randn(3, 3).cuda()
    y = torch.ones_like(x)
    print(f"Tensor on GPU: {x}")
    print(f"GPU tensor addition: {x + y}")

    try:
        # Test a simple xformers operation (e.g., flash attention)
        # This is a basic check; real usage would be more complex
        q = torch.randn(1, 16, 256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 16, 256, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 16, 256, 64, device="cuda", dtype=torch.float16)
        output = xops.memory_efficient_attention(q, k, v)
        print(f"xformers memory_efficient_attention test successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"xformers test failed: {e}")
        print("xformers might be installed but not correctly configured or your GPU might not support the specific operation.")

else:
    print("CUDA is not available. PyTorch will run on CPU.")

# Check torchvision (optional, but good practice)
try:
    import torchvision
    print(f"torchvision version: {torchvision.__version__}")
except ImportError:
    print("torchvision not found.")

print("Environment check complete.")