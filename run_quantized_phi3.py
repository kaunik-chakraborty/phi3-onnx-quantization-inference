#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import os

# Define the model ID and paths
model_id = "microsoft/Phi-3-mini-128k-instruct"
output_dir = "./phi3_quantized_model"
quantized_model_path = os.path.join(output_dir, "model.onnx") # Name of the quantized ONNX model

def run_quantized_model():
    """
    Loads the quantized ONNX model and allows interactive inference from the terminal.
    """
    print(f"\n--- Starting Interactive Inference with Quantized Phi-3.5 Mini Instruct ---")

    # 1. Check if the quantized model exists
    if not os.path.exists(quantized_model_path):
        print(f"Error: Quantized model not found at {quantized_model_path}")
        print("Please run 'quantize_phi3.py' first to quantize the model.")
        return

    # 2. Load the tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Load the quantized ONNX model
    print(f"Loading quantized model from {output_dir}...")
    try:
        # ORTModelForCausalLM can load directly from the directory containing model.onnx
        model = ORTModelForCausalLM.from_pretrained(output_dir)
        print("Quantized model loaded successfully.")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return

    print("Type your prompts below. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nUser: ")
            if prompt.lower() in ['exit', 'quit']:
                break

            # Prepare the prompt for the model
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

            print("Phi-3: ", end="")
            # Generate response
            # For ONNX Runtime, ensure the input_ids are on CPU
            outputs = model.generate(input_ids.to(model.device), max_new_tokens=500)

            # Decode and print the response
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # The output includes the prompt, so we need to extract only the new part
            # A simple way is to remove the input prompt from the generated text
            # This might need refinement for more complex chat templates
            response_text = text[len(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)):]
            print(response_text)

        except Exception as e:
            print(f"An error occurred during inference: {e}")
            print("Please ensure the model was quantized correctly and try again.")
            break

    print("--- Interactive Inference Session Ended ---")

if __name__ == "__main__":
    run_quantized_model()