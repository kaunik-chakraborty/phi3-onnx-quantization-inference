"""
Industry-level model quantization script with support for multiple models
and existing ONNX model quantization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelQuantizer:
    """Handles model downloading, conversion to ONNX, and quantization."""
    
    # Supported models configuration
    SUPPORTED_MODELS = {
        "phi3-mini": "microsoft/Phi-3-mini-128k-instruct",
        "phi3-medium": "microsoft/Phi-3-medium-128k-instruct", 
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    }
    
    QUANTIZATION_CONFIGS = {
        "arm64_dynamic": lambda: AutoQuantizationConfig.arm64(is_static=False, per_channel=False),
        "arm64_static": lambda: AutoQuantizationConfig.arm64(is_static=True, per_channel=True),
        "avx512": lambda: AutoQuantizationConfig.avx512(is_static=False, per_channel=False),
        "avx512_vnni": lambda: AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False),
    }

    def __init__(self, base_output_dir: str = "./quantized_models"):
        """Initialize the quantizer with base output directory."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.base_output_dir / "quantization.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def list_supported_models(self) -> None:
        """Display all supported models."""
        print("\n📋 Supported Models:")
        print("-" * 50)
        for key, model_id in self.SUPPORTED_MODELS.items():
            print(f"  {key:<15} -> {model_id}")
        print()

    def list_existing_onnx_models(self) -> List[str]:
        """List existing ONNX models in the base directory."""
        onnx_models = []
        for item in self.base_output_dir.iterdir():
            if item.is_dir():
                model_files = list(item.glob("*.onnx"))
                if model_files:
                    onnx_models.append(str(item))
        return onnx_models

    def get_model_output_dir(self, model_key: str) -> Path:
        """Get the output directory for a specific model."""
        return self.base_output_dir / f"{model_key}_quantized"

    def download_and_convert_to_onnx(self, model_id: str, output_dir: Path) -> bool:
        """Download model and convert to ONNX format."""
        try:
            self.logger.info(f"🔄 Loading model: {model_id}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to ONNX using ORTModelForCausalLM
            self.logger.info("🔄 Converting to ONNX format...")
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_id, 
                export=True, 
                device_map="cpu",
                torch_dtype=torch.float16
            )
            
            # Save ONNX model and tokenizer
            ort_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"✅ ONNX model saved to: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during ONNX conversion: {e}")
            return False

    def quantize_onnx_model(self, onnx_model_path: Path, quantization_type: str) -> bool:
        """Quantize an existing ONNX model."""
        try:
            self.logger.info(f"🔄 Quantizing model at: {onnx_model_path}")
            
            # Initialize quantizer
            quantizer = ORTQuantizer.from_pretrained(onnx_model_path)
            
            # Get quantization configuration
            if quantization_type not in self.QUANTIZATION_CONFIGS:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            qconfig = self.QUANTIZATION_CONFIGS[quantization_type]()
            
            # Create quantized output directory
            quantized_dir = onnx_model_path.parent / "quantized"
            quantized_dir.mkdir(exist_ok=True)
            
            # Perform quantization
            self.logger.info(f"🔄 Applying {quantization_type} quantization...")
            quantizer.quantize(
                save_dir=str(quantized_dir),
                quantization_config=qconfig,
                file_name="model.onnx"
            )
            
            self.logger.info(f"✅ Quantized model saved to: {quantized_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during quantization: {e}")
            return False

    def run_full_pipeline(self, model_key: str, quantization_type: str) -> bool:
        """Run the complete pipeline: download -> convert -> quantize."""
        if model_key not in self.SUPPORTED_MODELS:
            self.logger.error(f"❌ Unsupported model: {model_key}")
            return False
        
        model_id = self.SUPPORTED_MODELS[model_key]
        output_dir = self.get_model_output_dir(model_key)
        
        self.logger.info(f"🚀 Starting full pipeline for: {model_key}")
        
        # Step 1: Download and convert to ONNX
        if not self.download_and_convert_to_onnx(model_id, output_dir):
            return False
        
        # Step 2: Quantize the ONNX model
        if not self.quantize_onnx_model(output_dir, quantization_type):
            return False
        
        self.logger.info("🎉 Full pipeline completed successfully!")
        return True


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Industry-level model quantization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List supported models")
    
    # Quantize from scratch command
    quantize_parser = subparsers.add_parser("quantize", help="Download and quantize a model")
    quantize_parser.add_argument("model", choices=ModelQuantizer.SUPPORTED_MODELS.keys(),
                               help="Model to quantize")
    quantize_parser.add_argument("--quantization-type", 
                               choices=ModelQuantizer.QUANTIZATION_CONFIGS.keys(),
                               default="arm64_dynamic",
                               help="Quantization configuration")
    quantize_parser.add_argument("--output-dir", default="./quantized_models",
                               help="Base output directory")
    
    # Quantize existing ONNX command
    existing_parser = subparsers.add_parser("quantize-existing", 
                                          help="Quantize existing ONNX model")
    existing_parser.add_argument("onnx_path", help="Path to existing ONNX model directory")
    existing_parser.add_argument("--quantization-type",
                               choices=ModelQuantizer.QUANTIZATION_CONFIGS.keys(),
                               default="arm64_dynamic",
                               help="Quantization configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize quantizer
    quantizer = ModelQuantizer(getattr(args, 'output_dir', './quantized_models'))
    
    if args.command == "list":
        quantizer.list_supported_models()
        existing_models = quantizer.list_existing_onnx_models()
        if existing_models:
            print("📁 Existing ONNX Models:")
            print("-" * 50)
            for model_path in existing_models:
                print(f"  {model_path}")
            print()
    
    elif args.command == "quantize":
        success = quantizer.run_full_pipeline(args.model, args.quantization_type)
        sys.exit(0 if success else 1)
    
    elif args.command == "quantize-existing":
        onnx_path = Path(args.onnx_path)
        if not onnx_path.exists():
            print(f"❌ Error: ONNX model path does not exist: {onnx_path}")
            sys.exit(1)
        
        success = quantizer.quantize_onnx_model(onnx_path, args.quantization_type)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
