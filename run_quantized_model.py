"""
Industry-level quantized model runner with support for multiple models
and interactive model selection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


class QuantizedModelRunner:
    """Handles loading and running quantized ONNX models."""
    
    def __init__(self, base_models_dir: str = "./quantized_models"):
        """Initialize the model runner."""
        self.base_models_dir = Path(base_models_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

    def discover_quantized_models(self) -> Dict[str, Path]:
        """Discover all available quantized models."""
        models = {}
        
        if not self.base_models_dir.exists():
            self.logger.warning(f"Models directory does not exist: {self.base_models_dir}")
            return models
        
        for model_dir in self.base_models_dir.iterdir():
            if model_dir.is_dir():
                # Check for quantized subdirectory
                quantized_dir = model_dir / "quantized"
                if quantized_dir.exists() and (quantized_dir / "model.onnx").exists():
                    model_name = model_dir.name.replace("_quantized", "")
                    models[model_name] = quantized_dir
                # Check for direct ONNX model
                elif (model_dir / "model.onnx").exists():
                    model_name = model_dir.name.replace("_quantized", "")
                    models[model_name] = model_dir
        
        return models

    def list_available_models(self) -> None:
        """Display all available quantized models."""
        models = self.discover_quantized_models()
        
        if not models:
            print("❌ No quantized models found!")
            print(f"   Please run quantization first in: {self.base_models_dir}")
            return
        
        print("\n🤖 Available Quantized Models:")
        print("-" * 50)
        for i, (name, path) in enumerate(models.items(), 1):
            model_size = self._get_model_size(path / "model.onnx")
            print(f"  {i}. {name:<20} ({model_size})")
            print(f"     📍 {path}")
        print()

    def _get_model_size(self, model_path: Path) -> str:
        """Get human-readable model file size."""
        try:
            size_bytes = model_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.1f} TB"
        except:
            return "Unknown size"

    def load_model(self, model_path: Path, model_name: str) -> bool:
        """Load a quantized model and its tokenizer."""
        try:
            self.logger.info(f"🔄 Loading model: {model_name}")
            
            # Find the parent directory that contains the tokenizer
            tokenizer_path = model_path
            if "quantized" in str(model_path):
                tokenizer_path = model_path.parent
            
            # Load tokenizer
            self.logger.info("📚 Loading tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Load quantized model
            self.logger.info("🧠 Loading quantized ONNX model...")
            self.current_model = ORTModelForCausalLM.from_pretrained(model_path)
            
            self.current_model_name = model_name
            self.logger.info(f"✅ Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False

    def interactive_inference(self) -> None:
        """Run interactive inference session."""
        if not self.current_model or not self.current_tokenizer:
            print("❌ No model loaded! Please load a model first.")
            return
        
        print(f"\n🎯 Interactive Chat with {self.current_model_name}")
        print("=" * 60)
        print("💡 Tips:")
        print("  - Type 'exit' or 'quit' to end the session")
        print("  - Type 'clear' to clear conversation history") 
        print("  - Type 'info' to see model information")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'info':
                    self._show_model_info()
                    continue
                
                elif not user_input:
                    continue
                
                # Add user message to conversation
                conversation_history.append({"role": "user", "content": user_input})
                
                # Prepare input for model
                input_ids = self.current_tokenizer.apply_chat_template(
                    conversation_history, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                )
                
                print("🤖 Assistant: ", end="", flush=True)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.current_model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.current_tokenizer.eos_token_id
                    )
                
                # Decode response
                full_response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new response
                prompt_text = self.current_tokenizer.apply_chat_template(
                    conversation_history, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                response_text = full_response[len(prompt_text):].strip()
                
                print(response_text)
                
                # Add assistant response to conversation
                conversation_history.append({"role": "assistant", "content": response_text})
                
            except KeyboardInterrupt:
                print("\n\n🛑 Session interrupted by user.")
                break
            except Exception as e:
                self.logger.error(f"❌ Error during inference: {e}")
                print(f"\n❌ An error occurred: {e}")

    def _show_model_info(self) -> None:
        """Display current model information."""
        if not self.current_model:
            print("❌ No model loaded!")
            return
        
        print(f"\n📊 Model Information:")
        print(f"  🏷️  Name: {self.current_model_name}")
        print(f"  🧠 Type: Quantized ONNX")
        print(f"  📝 Vocab Size: {len(self.current_tokenizer)}")
        print(f"  🔧 Device: {self.current_model.device}")

    def select_and_load_model(self) -> bool:
        """Interactive model selection and loading."""
        models = self.discover_quantized_models()
        
        if not models:
            print("❌ No quantized models found!")
            return False
        
        self.list_available_models()
        
        while True:
            try:
                choice = input("🎯 Select a model (number or name): ").strip()
                
                # Try to parse as number
                try:
                    model_idx = int(choice) - 1
                    model_list = list(models.items())
                    if 0 <= model_idx < len(model_list):
                        model_name, model_path = model_list[model_idx]
                        break
                except ValueError:
                    pass
                
                # Try to match by name
                if choice in models:
                    model_name, model_path = choice, models[choice]
                    break
                
                print(f"❌ Invalid selection: {choice}")
                
            except KeyboardInterrupt:
                print("\n🛑 Selection cancelled.")
                return False
        
        return self.load_model(model_path, model_name)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Run quantized ONNX models interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--models-dir", default="./quantized_models",
                       help="Directory containing quantized models")
    parser.add_argument("--model", help="Specific model to load directly")
    parser.add_argument("--list", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = QuantizedModelRunner(args.models_dir)
    
    if args.list:
        runner.list_available_models()
        return
    
    if args.model:
        # Load specific model
        models = runner.discover_quantized_models()
        if args.model not in models:
            print(f"❌ Model not found: {args.model}")
            runner.list_available_models()
            sys.exit(1)
        
        if runner.load_model(models[args.model], args.model):
            runner.interactive_inference()
    else:
        # Interactive model selection
        if runner.select_and_load_model():
            runner.interactive_inference()


if __name__ == "__main__":
    main()
