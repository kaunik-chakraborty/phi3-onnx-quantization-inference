# Multi-Model ONNX Quantization and Inference

This repository contains Python scripts to demonstrate how to quantize various ONNX models using ONNX Runtime and Optimum, and then run interactive inference with the quantized models. The project is designed to be multi-model capable, not just for Phi-3.

## Requirements

Ensure you have the following Python packages installed globally:

```bash
pip install 'optimum[onnxruntime]'
pip install transformers
pip install torch
pip install onnx
```

## Files

-   `quantize_model.py`: Script to download and quantize various models.
-   `run_quantized_model.py`: Script to load the quantized model and run interactive inference.
-   `streamlit_app.py`: A Streamlit application for a user-friendly interface to select and interact with quantized models.

## Usage

1.  **Quantize a Model:**

    Run the `quantize_model.py` script to download a model and quantize it to ONNX format. This will create a directory containing the quantized model file.

    ```bash
    python3 quantize_model.py
    ```

2.  **Run Interactive Inference (CLI):**

    Once a model is quantized, run the `run_quantized_model.py` script. This will load the quantized model and start an interactive session where you can enter prompts.

    ```bash
    python3 run_quantized_model.py
    ```

    Type your prompt and press Enter. The model's response will be displayed. Type `exit` or `quit` to end the session.

3.  **Run Streamlit Application (GUI):**

    For a more user-friendly experience, run the Streamlit application. This will provide a web interface to select and interact with quantized models.

    ```bash
    python3 -m streamlit run streamlit_app.py
    ```

    ```bash
    python3 run_quantized_phi3.py
    ```

    Type your prompt and press Enter. The model's response will be displayed. Type `exit` or `quit` to end the session.

## Notes

-   The quantization is configured for ARM64 dynamic quantization, suitable for mobile deployment.
-   The scripts assume the required packages are installed globally.
-   Error handling is included in the scripts to catch common issues during loading and inference.