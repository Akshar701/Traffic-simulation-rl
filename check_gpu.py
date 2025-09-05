 #!/usr/bin/env python3
"""
GPU Diagnostic Script
====================

Check NVIDIA GPU setup and CUDA compatibility for optimal DQN training performance.
"""

import torch
import sys
import subprocess
import platform

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("🔍 Checking NVIDIA Driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].strip()
                    print(f"   Driver Version: {driver_version}")
                    break
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA driver may not be installed")
        return False

def check_cuda_installation():
    """Check CUDA installation (via nvcc if available, otherwise fallback to PyTorch)"""
    print("\n🔍 Checking CUDA Installation...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit is installed (via nvcc)")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].strip().split(',')[0]
                    print(f"   CUDA Version: {cuda_version}")
                    break
            return True
        else:
            print("⚠️ CUDA toolkit (nvcc) not found")
    except FileNotFoundError:
        print("⚠️ nvcc not found - CUDA toolkit may not be installed")

    # 🔄 Fallback: check PyTorch CUDA version
    if torch.version.cuda is not None:
        print("✅ CUDA runtime available through PyTorch")
        print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
        return True
    else:
        print("❌ No CUDA installation detected")
        return False


def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    print("\n🔍 Checking PyTorch CUDA Support...")
    
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        # Check each GPU
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("❌ PyTorch CUDA support not available")
        print("💡 Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

def check_system_info():
    """Check system information"""
    print("\n🔍 System Information...")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")

def test_gpu_performance():
    """Test GPU performance with a simple benchmark"""
    print("\n🚀 Testing GPU Performance...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping GPU test")
        return
    
    try:
        # Create test tensors
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # Warm up
        for _ in range(10):
            _ = torch.mm(x, y)
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(100):
            _ = torch.mm(x, y)
        torch.cuda.synchronize()
        end_time = time.time()
        
        ops_per_sec = 100 / (end_time - start_time)
        print(f"✅ GPU Performance: {ops_per_sec:.1f} matrix multiplications/second")
        
        # Memory test
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        print(f"   Memory Used: {memory_allocated:.1f} MB")
        
        # Clear memory
        del x, y
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

def check_optimization_settings():
    """Check and recommend optimization settings"""
    print("\n⚡ Optimization Settings...")
    
    if torch.cuda.is_available():
        # Check current settings
        print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if not torch.backends.cudnn.benchmark:
            print("   - Enable cuDNN benchmark for faster training")
        if torch.backends.cudnn.deterministic:
            print("   - Disable cuDNN deterministic for faster training (if reproducibility not needed)")
        
        # Memory optimization
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_total >= 8:
            print("   - Large GPU detected - consider increasing batch size")
        else:
            print("   - Smaller GPU - keep batch size moderate")
    else:
        print("   No GPU available for optimization")

def main():
    """Main diagnostic function"""
    print("🚀 GPU Diagnostic for DQN Training")
    print("=" * 50)
    
    # System info
    check_system_info()
    
    # Check components
    nvidia_ok = check_nvidia_driver()
    cuda_ok = check_cuda_installation()
    pytorch_ok = check_pytorch_cuda()
    
    # Performance test
    test_gpu_performance()
    
    # Optimization settings
    check_optimization_settings()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY:")
    
    if nvidia_ok and cuda_ok and pytorch_ok:
        print("✅ GPU setup is ready for optimal training!")
        print("\n🚀 Ready to train with:")
        print("   python3 train_dqn.py --episodes 500 --device cuda")
        print("   python3 train_dqn.py --episodes 500 --device cuda --batch-size 64")
    elif pytorch_ok:
        print("⚠️ PyTorch CUDA available but driver/toolkit issues detected")
        print("💡 Install NVIDIA driver and CUDA toolkit for optimal performance")
    else:
        print("❌ GPU training not available")
        print("💡 Install NVIDIA driver, CUDA toolkit, and CUDA-enabled PyTorch")
        print("   CPU training is still available but will be slower")
    
    print("\n📋 Next Steps:")
    if pytorch_ok:
        print("   1. Start GPU training: python3 train_dqn.py --episodes 100 --device cuda")
        print("   2. Monitor GPU memory: watch -n 1 nvidia-smi")
        print("   3. Optimize batch size based on GPU memory")
    else:
        print("   1. Install NVIDIA driver: https://www.nvidia.com/drivers")
        print("   2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("   3. Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("   4. Run this script again to verify setup")

if __name__ == "__main__":
    main()
