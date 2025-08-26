"""
Quick Model Preloader
Run this script to pre-load and cache all models for instant switching.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model_manager import initialize_model_cache
from src.utils.logging import app_logger

def main():
    print("🤖 Legal Document Analyzer - Model Preloader")
    print("=" * 50)
    print("This will download and cache all 6 AI models for instant switching.")
    print("⚠️  This requires significant disk space (~15-20 GB) and internet bandwidth.")
    print()
    
    response = input("Continue with model preloading? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Preloading cancelled.")
        return
    
    print("\n🚀 Starting model preloading...")
    print("This may take 30-60 minutes depending on your internet connection.")
    print()
    
    try:
        success = initialize_model_cache()
        
        if success:
            print("\n✅ SUCCESS! All models have been preloaded and cached.")
            print("🚀 Your users can now switch between models instantly!")
            print("\nRun 'python manage_models.py info' to see cache details.")
        else:
            print("\n⚠️  Some models failed to preload. Check the logs for details.")
            print("You can retry with 'python preload_models.py'")
            
    except KeyboardInterrupt:
        print("\n⏹️  Preloading interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during preloading: {e}")
        print("Check the logs for more details.")

if __name__ == "__main__":
    main()
