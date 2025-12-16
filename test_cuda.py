import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA device capability:", torch.cuda.get_device_capability(0))
    
    # Test CUDA operation
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    
    # CPU operation
    import time
    start = time.time()
    z_cpu = torch.matmul(x, y)
    cpu_time = time.time() - start
    print(f"CPU matmul time: {cpu_time:.4f} seconds")
    
    # CUDA operation
    x_cuda = x.to("cuda")
    y_cuda = y.to("cuda")
    
    start = time.time()
    z_cuda = torch.matmul(x_cuda, y_cuda)
    torch.cuda.synchronize()  # Wait for CUDA operations to complete
    cuda_time = time.time() - start
    print(f"CUDA matmul time: {cuda_time:.4f} seconds")
    print(f"CUDA speedup: {cpu_time / cuda_time:.2f}x")
    
    # Verify results are the same
    z_cpu_from_cuda = z_cuda.to("cpu")
    print(f"Results match: {torch.allclose(z_cpu, z_cpu_from_cuda, atol=1e-5)}")
else:
    print("CUDA is not available. Using CPU only.")
