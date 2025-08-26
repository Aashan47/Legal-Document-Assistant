"""
Model Cache Management Script
Use this script to pre-load and manage cached models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.models.model_manager import get_model_manager, initialize_model_cache
from src.utils.logging import app_logger


def preload_models():
    """Pre-load all available models."""
    print("🤖 Starting model pre-loading process...")
    print("⚠️  This will download and cache all models. It may take significant time and disk space.")
    
    # Confirm action
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Operation cancelled.")
        return False
    
    # Initialize and preload
    success = initialize_model_cache()
    
    if success:
        print("✅ All models successfully pre-loaded and cached!")
        show_cache_info()
        return True
    else:
        print("❌ Some models failed to pre-load. Check logs for details.")
        return False


def show_cache_info():
    """Show information about cached models."""
    manager = get_model_manager()
    info = manager.get_cache_info()
    
    print("\n📊 Model Cache Information:")
    print(f"├─ Total models available: {info['total_available']}")
    print(f"├─ Models cached: {info['total_cached']}")
    print(f"├─ Total cache size: {info['total_size_gb']:.2f} GB")
    print(f"└─ Current model: {info.get('current_model', 'None')}")
    
    if info['cached_models']:
        print("\n📦 Cached Models:")
        for model in info['cached_models']:
            print(f"  ├─ {model['model_name']}")
            print(f"  │  ├─ Model ID: {model['model_id']}")
            print(f"  │  └─ Size: {model['size_mb']:.1f} MB")


def clear_cache():
    """Clear all cached models."""
    print("🗑️  This will delete all cached models.")
    response = input("Are you sure? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Operation cancelled.")
        return
    
    manager = get_model_manager()
    cache_dir = manager.cache_dir
    
    try:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cache cleared successfully!")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")


def test_model_switching():
    """Test model switching performance."""
    from src.models.model_config import AVAILABLE_MODELS
    
    print("🧪 Testing model switching performance...")
    manager = get_model_manager()
    
    test_question = "What are the key terms in this contract?"
    test_context = ["This is a sample legal document with various clauses and terms."]
    
    for model_key, config in list(AVAILABLE_MODELS.items())[:3]:  # Test first 3 models
        model_id = config.model_id
        print(f"\n🔄 Testing {config.name} ({model_id})...")
        
        try:
            import time
            start_time = time.time()
            
            # Load model
            if manager.load_model_for_inference(model_id):
                load_time = time.time() - start_time
                print(f"  ├─ Load time: {load_time:.2f}s")
                
                # Generate response
                start_time = time.time()
                response = manager.generate_response(test_question, test_context, model_id)
                response_time = time.time() - start_time
                
                print(f"  ├─ Response time: {response_time:.2f}s")
                print(f"  ├─ Confidence: {response['confidence']:.2f}")
                print(f"  └─ Answer length: {len(response['answer'])} chars")
            else:
                print("  └─ ❌ Failed to load model")
                
        except Exception as e:
            print(f"  └─ ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Model Cache Management")
    parser.add_argument('action', choices=['preload', 'info', 'clear', 'test'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'preload':
        preload_models()
    elif args.action == 'info':
        show_cache_info()
    elif args.action == 'clear':
        clear_cache()
    elif args.action == 'test':
        test_model_switching()


if __name__ == "__main__":
    main()
