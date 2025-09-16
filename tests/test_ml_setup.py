#!/usr/bin/env python3
"""
Test script for RTX 4090 ML Environment Setup
Tests PyTorch, CUDA, and Hugging Face integration
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

def test_pytorch_cuda():
    """Test PyTorch and CUDA availability"""
    print("🔍 Testing PyTorch and CUDA Setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor operations on GPU
        print("\n🧪 Testing GPU tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"GPU tensor operation successful! Result shape: {z.shape}")
        
        # Test memory info
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        return True
    else:
        print("❌ CUDA not available - drivers may need reboot")
        return False

def test_huggingface():
    """Test Hugging Face integration"""
    print("\n🤗 Testing Hugging Face Integration...")
    
    try:
        # Test tokenizer and model loading
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test tokenization
        text = "Polkadot is a next-generation blockchain protocol"
        inputs = tokenizer(text, return_tensors="pt")
        
        print(f"Tokenization successful! Input shape: {inputs['input_ids'].shape}")
        
        # Test model inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"Model inference successful! Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face test failed: {e}")
        return False

def test_ml_packages():
    """Test ML package imports"""
    print("\n📦 Testing ML Package Imports...")
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('plotly', 'plotly'),
        ('xgboost', 'xgb'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('huggingface_hub', 'hf_hub')
    ]
    
    success_count = 0
    for package, alias in packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    print(f"\n📊 Package test results: {success_count}/{len(packages)} packages working")
    return success_count == len(packages)

def test_polkadot_data_access():
    """Test access to Polkadot data"""
    print("\n🔗 Testing Polkadot Data Access...")
    
    try:
        # Check if we can access the existing Polkadot data files
        import os
        import json
        
        data_files = [
            'polkadot_metrics.db',
            'l2_networks_analysis.json',
            'l2_dashboard_data.json'
        ]
        
        for file in data_files:
            if os.path.exists(file):
                print(f"✅ Found {file}")
                if file.endswith('.json'):
                    with open(file, 'r') as f:
                        data = json.load(f)
                        print(f"   - Contains {len(data)} entries")
            else:
                print(f"❌ Missing {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Polkadot data test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 RTX 4090 ML Environment Test Suite")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("ML Packages", test_ml_packages),
        ("Hugging Face", test_huggingface),
        ("Polkadot Data", test_polkadot_data_access)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ML environment is ready!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return results

if __name__ == "__main__":
    main()
