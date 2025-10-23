#!/usr/bin/env python3
"""
[EDU] Understand Power Sampling - Educational Demo

This file demonstrates WHAT Power Sampling does and WHY it improves LLM reasoning.
Perfect for understanding the algorithm step by step.

Author: Educational demo by Craicek (BitMakerMan)
Based on: Original Power Sampling by aakaran
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global parameter for model size
params_for_small_model = True

try:
    from power_sampling import load_model_and_tokenizer, power_sample
    print("[OK] Successfully imported Power Sampling modules")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"[DEMO] {title}")
    print(f"{'='*60}")

def print_comparison(original, improved, prompt):
    """Print side-by-side comparison"""
    print(f"\n[PROMPT]: {prompt}")
    print("-" * 60)

    print(f"[STANDARD GENERATION]:")
    print(f"   {original}")
    print()

    print(f"[POWER SAMPLING (Improved)]:")
    print(f"   {improved}")
    print()

    print("[IMPROVEMENTS]:")

    # Simple analysis of improvements
    if len(improved) > len(original):
        print("   + More detailed response")
    if "because" in improved.lower() or "therefore" in improved.lower():
        print("   + Better logical connections")
    if "step" in improved.lower() or "first" in improved.lower():
        print("   + Structured reasoning")
    if improved.count('.') > original.count('.'):
        print("   + More complete sentences")

    print("-" * 60)

def demonstrate_basic_generation(model, tokenizer):
    """Show basic LLM generation without Power Sampling"""
    print_header("1. Standard LLM Generation (Without Power Sampling)")

    prompt = "What is artificial intelligence?"
    print(f"Prompt: {prompt}")
    print("This shows how a normal language model generates text...")

    # Standard generation without attention mask (to avoid error)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,  # Increased for longer responses
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )

    standard_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the original prompt from response
    if standard_response.startswith(prompt):
        standard_response = standard_response[len(prompt):].strip()

    print(f"Response: {standard_response}")

    return standard_response

def demonstrate_power_sampling(model, tokenizer):
    """Show Power Sampling in action"""
    print_header("2. Power Sampling in Action")

    prompt = "What is artificial intelligence?"
    print(f"Same prompt: {prompt}")
    print("Power Sampling applies Metropolis-Hastings to improve coherence...")

    # Use conservative parameters for small model
    print("\n[PARAMS] Testing conservative Power Sampling parameters:")
    print("Note: Using smaller alpha and fewer steps for this 125M parameter model")

    for alpha in [1.5, 2.0, 2.5]:
        print(f"\n--- Alpha = {alpha} (Conservative Sharpening) ---")

        try:
            response = power_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=alpha,
                steps=3,  # More steps for better development
                block_size=20,  # Larger blocks for more context
                max_len=120,  # Longer responses
                show_progress=False,
                temperature=0.8  # Lower temperature
            )

            # Clean up response
            response = response.strip()
            if len(response) > 150:
                response = response[:150] + "..."
            print(f"Response: {response}")

        except Exception as e:
            print(f"[ERROR] Error with alpha={alpha}: {e}")
            print("Trying with standard generation...")

            # Fallback to standard generation
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            standard_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Fallback response: {standard_response}")

def demonstrate_step_by_step(model, tokenizer):
    """Show how Power Sampling works step by step"""
    print_header("3. How Power Sampling Works - Step by Step")

    prompt = "Explain why the sky is blue"
    print(f"Prompt: {prompt}")
    print("\n[PROCESS] Power Sampling Process:")
    print("1. Generate initial text")
    print("2. Divide text into blocks")
    print("3. Propose alternatives for each block")
    print("4. Accept/reject based on probability improvement")
    print("5. Repeat for multiple iterations")

    # Run with conservative parameters and progress
    print(f"\nRunning Power Sampling with visible steps...")
    print("Using conservative parameters for small model...")

    try:
        response = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=2.0,      # Conservative alpha
            steps=3,        # Fewer steps
            block_size=16,  # Smaller blocks
            max_len=80,     # Shorter response
            show_progress=True,
            temperature=0.7
        )

        print(f"\n[FINAL] Final Improved Response:")
        print(f"   {response}")

    except Exception as e:
        print(f"[ERROR] Error in step-by-step demo: {e}")
        print("Demonstrating with standard generation instead...")

        # Fallback demonstration
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[STANDARD] Standard response: {response}")

def compare_different_prompts(model, tokenizer):
    """Compare Power Sampling on different types of prompts"""
    print_header("4. Power Sampling on Different Question Types")

    # Use simpler, more concrete prompts for small model
    prompts = [
        "What is a cat?",
        "Why do birds fly?",
        "How do cars move?",
        "What makes plants grow?"
    ]

    print("Note: Using simple prompts suitable for 125M parameter model")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[TEST {i}]: {prompt}")

        # Get standard response
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,  # Longer for better responses
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Reduce repetition
                )
            standard = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from response
            if standard.startswith(prompt):
                standard = standard[len(prompt):].strip()

        except Exception as e:
            print(f"[ERROR] Error generating standard response: {e}")
            standard = f"Error generating response for: {prompt}"

        # Get Power Sampling response with better parameters
        try:
            improved = power_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=1.8,      # Conservative
                steps=3,        # More steps for development
                block_size=16,  # Medium blocks
                max_len=80,     # Longer responses
                show_progress=False,
                temperature=0.8
            )
        except Exception as e:
            print(f"[ERROR] Error in Power Sampling: {e}")
            print("Using standard generation as fallback...")
            improved = standard

        print_comparison(standard, improved, prompt)

def explain_theory():
    """Explain the theory behind Power Sampling"""
    print_header("5. The Theory Behind Power Sampling")

    print("""
[THEORY] What is Power Sampling?

Power Sampling is an algorithm that improves LLM reasoning using the
Metropolis-Hastings method (a technique from computational statistics).

[HOW] How it Works:

1. INITIAL GENERATION
   - Generate text normally with the LLM
   - This gives us a starting point

2. BLOCK RESAMPLING
   - Split the text into smaller blocks
   - For each block, propose alternative text
   - Calculate probability scores for both original and new text

3. METROPOLIS-HASTINGS DECISION
   - Accept the new text IF it improves overall probability
   - Sometimes accept worse text to explore possibilities
   - This helps avoid getting stuck in local optima

4. ITERATION
   - Repeat the process multiple times
   - Each iteration typically improves the text
   - More iterations = better quality (but slower)

[WHY] Why it Works:

- **Focus**: The "alpha" parameter sharpens probabilities, making better text more likely
- **Exploration**: Sometimes accepts worse options to find better solutions
- **Coherence**: Evaluates text in context, not just word by word
- **Iteration**: Multiple rounds of refinement

[PARAMS] Parameters:

- **Alpha (2.0-6.0)**: Higher = more focused on high-probability text
- **Steps (1-10)**: Number of refinement iterations
- **Block Size (16-128)**: Size of text chunks to resample
- **Max Length**: Maximum response length

[RESULT] The Result: More coherent, logical, and well-structured text!
""")

def main():
    """Main demonstration function"""
    print_header("[EDU] Understanding Power Sampling - Educational Demo")
    print("This demo will show you exactly what Power Sampling does and how it works!")
    print("Based on original work by aakaran, educational version by Craicek")

    # Model options with better performance
    model_options = {
        "1": ("gpt2", "GPT-2 (124M) - Fast, basic reasoning"),
        "2": ("EleutherAI/gpt-neo-125M", "GPT-Neo 125M - Better logic"),
        "3": ("microsoft/DialoGPT-medium", "DialoGPT Medium (345M) - Conversational"),
        "4": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B - Small but capable"),
        "5": ("bigscience/bloom-560m", "BLOOM-560M - Good reasoning"),
        "6": ("facebook/opt-1.3b", "OPT-1.3B - Open Pretrained"),
        "7": "Custom model name from HuggingFace"
    }

    print("\n[MODELS] Available Models:")
    for key, (model_name, description) in list(model_options.items())[:-1]:
        print(f"  {key}. {description}")
    print(f"  7. Enter custom HuggingFace model name")

    choice = input("\nSelect model (1-7): ").strip()

    if choice in model_options and choice != "7":
        model_name, description = model_options[choice]
        print(f"\n[OK] Selected: {description}")
    elif choice == "7":
        model_name = input("Enter HuggingFace model name: ").strip()
        print(f"\n[OK] Selected custom model: {model_name}")
    else:
        print("[ERROR] Invalid choice, using default model...")
        model_name = "EleutherAI/gpt-neo-125M"
        description = "GPT-Neo 125M - Default model"

    # Check for local model first
    base_model_path = os.path.join(os.path.dirname(__file__), '..', 'models--EleutherAI--gpt-neo-125M')
    model_path = os.path.join(base_model_path, 'snapshots', '21def0189f5705e2521767faed922f1f15e7d7db')

    use_local = False
    if model_name == "EleutherAI/gpt-neo-125M" and os.path.exists(model_path):
        use_local = True
        print(f"\n[LOCAL] Using local model: {model_path}")
        model_path_to_use = model_path
    else:
        print(f"\n[DOWNLOAD] Downloading from HuggingFace: {model_name}")
        print("[INFO] This may take a few minutes for first-time download...")
        model_path_to_use = model_name

    try:
        print(f"[LOADING] Loading model...")
        model, tokenizer = load_model_and_tokenizer(model_path_to_use)

        # Adjust parameters based on model size
        if "124M" in description or "gpt-neo-125M" in model_name:
            print("[INFO] Detected small model - using conservative parameters")
            global params_for_small_model
            params_for_small_model = True
        else:
            print("[INFO] Detected medium/large model - using standard parameters")
            params_for_small_model = False

        print("[OK] Model loaded successfully!")

        # Explain the theory first
        explain_theory()

        # Show standard generation
        standard_response = demonstrate_basic_generation(model, tokenizer)

        # Show Power Sampling
        demonstrate_power_sampling(model, tokenizer)

        # Step by step demo
        demonstrate_step_by_step(model, tokenizer)

        # Compare different prompts
        compare_different_prompts(model, tokenizer)

        print_header("[SUMMARY] Summary")
        print("""
[SUCCESS] You've seen how Power Sampling works!

Key Takeaways:
1. Power Sampling improves LLM text quality
2. Uses Metropolis-Hastings algorithm for refinement
3. Works by iteratively improving text blocks
4. Parameters control quality vs speed trade-offs
5. Results in more coherent and logical responses

[INFO] For more information:
- Original repository: https://github.com/aakaran/reasoning-with-sampling
- Educational demo: https://github.com/BitMakerMan/PowerSampling

[INFO] Try it yourself with different prompts and parameters!
""")

    except Exception as e:
        print(f"[ERROR] Error during demonstration: {e}")
        print("This is normal - LLM generation can sometimes fail.")
        print("Try running the demo again!")

if __name__ == "__main__":
    main()