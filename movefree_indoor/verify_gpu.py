"""
GPU and CUDA Verification for MoveFree Training
Run this BEFORE training to ensure everything is configured correctly
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_cuda_setup():
    """Comprehensive CUDA verification"""
    logger.info("=" * 60)
    logger.info("🔍 CUDA & GPU VERIFICATION")
    logger.info("=" * 60)
    
    # 1. Check PyTorch installation
    logger.info(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # 2. Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"🎮 CUDA Available: {cuda_available}")
    
    if not cuda_available:
        logger.error("❌ CUDA NOT AVAILABLE!")
        logger.error("Possible reasons:")
        logger.error("  1. NVIDIA drivers not installed")
        logger.error("  2. CUDA toolkit not installed")
        logger.error("  3. PyTorch CPU version installed (not GPU version)")
        logger.error("\nTo fix:")
        logger.error("  pip uninstall torch torchvision")
        logger.error("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # 3. CUDA version
    logger.info(f"🔧 CUDA Version: {torch.version.cuda}")
    
    # 4. cuDNN version (if available)
    if torch.backends.cudnn.is_available():
        logger.info(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")
        logger.info(f"🔧 cuDNN Enabled: {torch.backends.cudnn.enabled}")
    
    # 5. Number of GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"\n🎮 Number of GPUs: {num_gpus}")
    
    # 6. GPU details
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        logger.info(f"\n   GPU {i}: {gpu_name}")
        logger.info(f"   Memory: {gpu_memory:.2f} GB")
        
        # Memory check
        if gpu_memory < 4:
            logger.warning(f"   ⚠️ GPU {i} has less than 4GB memory. Training may be slow.")
    
    # 7. Test GPU computation
    logger.info("\n🧪 Testing GPU computation...")
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform computation
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        logger.info(f"✅ GPU computation successful!")
        logger.info(f"   Matrix multiplication (1000x1000): {elapsed*1000:.2f} ms")
        
    except Exception as e:
        logger.error(f"❌ GPU computation failed: {e}")
        return False
    
    # 8. Check GPU memory
    logger.info(f"\n💾 GPU Memory Status:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        logger.info(f"   GPU {i}:")
        logger.info(f"      Allocated: {allocated:.2f} GB")
        logger.info(f"      Reserved:  {reserved:.2f} GB")
        logger.info(f"      Total:     {total:.2f} GB")
        logger.info(f"      Free:      {total - reserved:.2f} GB")
    
    # 9. Recommended settings for RTX 3070 Ti
    logger.info("\n" + "=" * 60)
    logger.info("📊 RECOMMENDED SETTINGS FOR RTX 3070 Ti (8GB)")
    logger.info("=" * 60)
    logger.info("For YOLOv8n (nano) - Raspberry Pi target:")
    logger.info("   batch_size: 32-64")
    logger.info("   imgsz: 416")
    logger.info("   workers: 8")
    logger.info("")
    logger.info("For YOLOv8s (small) - Jetson target:")
    logger.info("   batch_size: 32-48")
    logger.info("   imgsz: 640")
    logger.info("   workers: 8")
    logger.info("")
    logger.info("⚠️ If you get 'CUDA out of memory' errors:")
    logger.info("   - Reduce batch_size (try 16, then 8)")
    logger.info("   - Disable cache: cache=False")
    logger.info("   - Enable mixed precision: amp=True")
    
    logger.info("\n✅ GPU verification complete!")
    logger.info("You're ready to train! 🚀")
    
    return True


def benchmark_gpu():
    """Benchmark GPU performance"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. Cannot benchmark.")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("⚡ GPU PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    import time
    import numpy as np
    
    # Test different operations
    tests = [
        ("Matrix Multiplication (1000x1000)", lambda: torch.matmul(
            torch.randn(1000, 1000, device='cuda'),
            torch.randn(1000, 1000, device='cuda')
        )),
        ("Convolution (3x224x224, 64 filters)", lambda: torch.nn.functional.conv2d(
            torch.randn(1, 3, 224, 224, device='cuda'),
            torch.randn(64, 3, 3, 3, device='cuda')
        )),
        ("Batch Normalization", lambda: torch.nn.functional.batch_norm(
            torch.randn(32, 256, 56, 56, device='cuda'),
            torch.randn(256, device='cuda'),
            torch.randn(256, device='cuda'),
            training=True
        )),
    ]
    
    for test_name, test_func in tests:
        # Warm-up
        for _ in range(5):
            test_func()
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            test_func()
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        
        logger.info(f"\n{test_name}:")
        logger.info(f"   Average: {avg_time:.2f} ms ± {std_time:.2f} ms")


if __name__ == "__main__":
    success = verify_cuda_setup()
    
    if success:
        # Optional: Run benchmark
        response = input("\nRun GPU performance benchmark? (y/n): ")
        if response.lower() == 'y':
            benchmark_gpu()
    else:
        logger.error("\n❌ Please fix CUDA issues before training!")