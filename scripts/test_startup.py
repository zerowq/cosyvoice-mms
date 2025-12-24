#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

def test_startup():
    print("Testing imports...")
    try:
        from src.config import config
        print(f"Config loaded. Device: {config.device}")
        
        # Test Engine imports (without loading heavy models)
        from src.engines.mms_engine import MMSEngine
        print("MMS Engine imported successfully.")
        
        # We don't import CosyVoice here because it requires submodules to be present
        if (ROOT_DIR / "CosyVoice").exists():
             print("CosyVoice folder exists.")
             
        print("\n✅ Basic startup check passed!")
        
    except Exception as e:
        print(f"\n❌ Startup check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_startup()
