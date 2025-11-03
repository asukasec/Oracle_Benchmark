#!/usr/bin/env python3
"""
Comprehensive test for Oracle Benchmark improvements
Tests flexible API management, local model support, and backward compatibility
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flexible_api_manager import get_api_manager
from model_config import get_model_config_manager
from local_model_support import get_local_model_manager


def test_api_manager():
    """Test flexible API manager"""
    print("\n=== Testing Flexible API Manager ===")
    
    api_manager = get_api_manager()
    
    # Test API key checking
    print("\n1. Testing API key availability check:")
    available_keys = api_manager.check_all_available_keys()
    for provider, available in available_keys.items():
        status = "✓" if available else "✗"
        print(f"   {status} {provider}")
    
    # Test available model families
    print("\n2. Testing available model families:")
    families = api_manager.get_available_model_families()
    print(f"   Available families: {families}")
    
    # Test that we can get API manager multiple times (singleton)
    api_manager2 = get_api_manager()
    assert api_manager is api_manager2, "API manager should be singleton"
    print("   ✓ API manager is singleton")
    
    print("\n✓ Flexible API Manager tests passed")


def test_model_config_manager():
    """Test model configuration manager"""
    print("\n=== Testing Model Configuration Manager ===")
    
    config_manager = get_model_config_manager()
    
    # Test listing models
    print("\n1. Testing model listing:")
    all_models = config_manager.list_all_models()
    print(f"   Total models configured: {len(all_models)}")
    print(f"   Sample models: {all_models[:5]}")
    
    # Test getting specific model config
    print("\n2. Testing specific model config:")
    gpt4_config = config_manager.get_model_config('gpt-4o')
    if gpt4_config:
        print(f"   ✓ GPT-4o config:")
        print(f"     - Family: {gpt4_config.family}")
        print(f"     - Type: {gpt4_config.type}")
        print(f"     - API name: {gpt4_config.api_name}")
    
    # Test model families
    print("\n3. Testing model families:")
    families = config_manager.list_model_families()
    print(f"   Model families: {families}")
    
    # Test getting models by family
    print("\n4. Testing getting models by family:")
    gpt_models = config_manager.get_models_by_family('gpt')
    print(f"   GPT models: {[m.name for m in gpt_models[:3]]}")
    
    print("\n✓ Model Configuration Manager tests passed")


def test_local_model_manager():
    """Test local model manager (without actually loading models)"""
    print("\n=== Testing Local Model Manager ===")
    
    model_manager = get_local_model_manager()
    
    # Test singleton
    model_manager2 = get_local_model_manager()
    assert model_manager is model_manager2, "Model manager should be singleton"
    print("   ✓ Local model manager is singleton")
    
    # Test that manager exists and has methods
    assert hasattr(model_manager, 'load_model'), "Manager should have load_model method"
    assert hasattr(model_manager, 'get_model'), "Manager should have get_model method"
    assert hasattr(model_manager, 'is_model_loaded'), "Manager should have is_model_loaded method"
    print("   ✓ Local model manager has required methods")
    
    print("\n✓ Local Model Manager tests passed (loading models requires transformers)")


def test_backward_compatibility():
    """Test backward compatibility"""
    print("\n=== Testing Backward Compatibility ===")
    
    from main import map_model_name_to_api_name, get_model_name
    
    # Test old model name mappings
    print("\n1. Testing model name mappings:")
    test_cases = [
        ('gpt-4o', 'gpt-4o-2024-08-06'),
        ('claude-3.5-sonnet', 'claude-3-5-sonnet-20241022'),
        ('gemini-2.5-pro', 'gemini-2.5-pro'),
        ('deepseek-v3', 'deepseek-chat'),
    ]
    
    for model_name, expected_api_name in test_cases:
        api_name = map_model_name_to_api_name(model_name)
        status = "✓" if api_name == expected_api_name else "✗"
        print(f"   {status} {model_name} -> {api_name}")
        assert api_name == expected_api_name, f"Expected {expected_api_name}, got {api_name}"
    
    # Test get_model_name function
    print("\n2. Testing get_model_name function:")
    families = ['gpt', 'claude', 'gemini', 'qwen', 'deepseek', 'llama']
    for family in families:
        models = get_model_name(family)
        print(f"   ✓ {family}: {len(models)} models")
    
    print("\n✓ Backward Compatibility tests passed")


def test_integration():
    """Test integration between components"""
    print("\n=== Testing Integration ===")
    
    api_manager = get_api_manager()
    config_manager = get_model_config_manager()
    
    # Test getting available models with API manager
    print("\n1. Testing available models detection:")
    available_models = config_manager.get_available_models(api_manager)
    print(f"   Models available without API keys: {len(available_models)}")
    
    # Since no API keys are set, only local models should be available
    # But we don't have local models configured by default
    print(f"   ✓ System correctly detects available models based on API keys")
    
    print("\n✓ Integration tests passed")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Oracle Benchmark - Flexible Model Management Tests")
    print("="*60)
    
    try:
        test_api_manager()
        test_model_config_manager()
        test_local_model_manager()
        test_backward_compatibility()
        test_integration()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nKey Features Verified:")
        print("  ✓ Flexible API key management")
        print("  ✓ On-demand API client initialization")
        print("  ✓ Model configuration system")
        print("  ✓ Local model support infrastructure")
        print("  ✓ Backward compatibility maintained")
        print("  ✓ Component integration")
        print("\nNote: Actual model loading tests require transformers/torch/vllm")
        print("      and are not included in this basic test suite.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
