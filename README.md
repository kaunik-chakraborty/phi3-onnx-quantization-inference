# Phi-3.5 Mini Instruct ONNX Quantization and Inference

This repository contains two Python scripts to demonstrate how to quantize the Microsoft Phi-3.5 Mini Instruct model using ONNX Runtime and Optimum, and then run interactive inference with the quantized model.

## Requirements

Ensure you have the following Python packages installed globally:

```bash
pip install 'optimum[onnxruntime]'
pip install transformers
pip install torch
pip install onnx
```

## Files

-   `quantize_phi3.py`: Script to download and quantize the Phi-3.5 Mini Instruct model.
-   `run_quantized_phi3.py`: Script to load the quantized model and run interactive inference.

## Usage

1.  **Quantize the Model:**

    Run the `quantize_phi3.py` script to download the model and quantize it to ONNX format for ARM64 dynamic quantization. This will create a directory named `phi3_quantized_model` containing the quantized model file (`model.onnx`).

    ```bash
    python3 quantize_phi3.py
    ```

2.  **Run Interactive Inference:**

    Once the model is quantized, run the `run_quantized_phi3.py` script. This will load the quantized model and start an interactive session where you can enter prompts.

    ```bash
    python3 run_quantized_phi3.py
    ```

    Type your prompt and press Enter. The model's response will be displayed. Type `exit` or `quit` to end the session.

## Notes

-   The quantization is configured for ARM64 dynamic quantization, suitable for mobile deployment.
-   The scripts assume the required packages are installed globally.
-   Error handling is included in the scripts to catch common issues during loading and inference.