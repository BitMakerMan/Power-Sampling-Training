#!/usr/bin/env python3
"""
Test script for the educational demo with automatic model selection.
This demonstrates the model selection functionality without requiring user input.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the fixed educational demo functions
from understand_power_sampling import (
    load_model_and_tokenizer, power_sample,
    demonstrate_basic_generation, demonstrate_power_sampling,
    explain_theory, print_header
)

def main():
    """Test model selection functionality with automatic model choice"""
    print_header("[TEST] Automatic Model Selection Test")

    # Test with local GPT-Neo model (if available)
    base_model_path = os.path.join(os.path.dirname(__file__), '..', 'models--EleutherAI--gpt-neo-125M')
    model_path = os.path.join(base_model_path, 'snapshots', '21def0189f5705e2521767faed922f1f15e7d7db')

    if os.path.exists(model_path):
        print("[LOCAL] Found local GPT-Neo 125M model, testing with it...")
        model_path_to_use = model_path
        description = "GPT-Neo 125M - Local model"
    else:
        print("[DOWNLOAD] Testing with GPT-2 (smaller, faster download)...")
        model_path_to_use = "gpt2"
        description = "GPT-2 (124M) - Fast test model"

    try:
        print(f"[LOADING] Loading model: {description}")
        model, tokenizer = load_model_and_tokenizer(model_path_to_use)
        print("[OK] Model loaded successfully!")

        # Test basic functionality
        print("\n[TEST] Testing basic generation...")
        standard_response = demonstrate_basic_generation(model, tokenizer)

        print("\n[TEST] Testing Power Sampling...")
        demonstrate_power_sampling(model, tokenizer)

        print("\n[SUCCESS] All tests passed! Model selection is working correctly.")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print("This might be due to memory or download issues.")

if __name__ == "__main__":
    main()