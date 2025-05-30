#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import os

# Define the model ID
model_id = "microsoft/Phi-3-mini-128k-instruct"
output_dir = "./phi3_quantized_model"
quantized_model_path = os.path.join(output_dir, "phi3_quantized.onnx")

def quantize_model():
    """
    Downloads the Phi-3.5 Mini Instruct model, quantizes it to ONNX format
    using dynamic quantization for ARM64, and saves the quantized model.
    """
    print(f"\n--- Starting Model Quantization for {model_id} ---")

    # 1. Load the tokenizer and model
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
        print("Tokenizer and model loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # 2. Create ONNX Runtime model
    print("Creating ONNX Runtime model...")
    try:
        # Export the model to ONNX format first
        onnx_model_path = os.path.join(output_dir, "phi3_unquantized.onnx")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Optimum's ORTModelForCausalLM handles the export internally when saving
        # We need to ensure the model is in eval mode for export
        model.eval()
        
        # Save the model to ONNX format. This step also handles the conversion.
        # The `for_quantization=True` argument prepares the model for quantization.
        ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True, device_map="cpu")
        ort_model.save_pretrained(output_dir)
        print(f"Unquantized ONNX model saved to {output_dir}")

    except Exception as e:
        print(f"Error creating ORTModelForCausalLM or saving unquantized model: {e}")
        return

    # 3. Initialize the quantizer
    print("Initializing quantizer...")
    try:
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        print("Quantizer initialized.")
    except Exception as e:
        print(f"Error initializing quantizer: {e}")
        return

    # 4. Define quantization configuration for ARM64 dynamic quantization
    # Using 'arm64' for target_platform and 'dynamic' for quantization_mode
    print("Defining quantization configuration...")
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    print("Quantization configuration defined.")

    # 5. Quantize the model
    print("Quantizing the model...")
    try:
        # The quantizer expects the path to the unquantized ONNX model
        # which is now located in output_dir after ort_model.save_pretrained()
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=qconfig,
            file_name="model.onnx" # The default ONNX file name saved by ORTModelForCausalLM
        )
        print(f"Model quantized and saved to {output_dir}")
    except Exception as e:
        print(f"Error during quantization: {e}")
        return

    print("--- Model Quantization Complete ---")

if __name__ == "__main__":
    quantize_model()