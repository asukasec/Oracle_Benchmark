"""
Local Model Support for Oracle Benchmark
Provides integration with Hugging Face transformers, vLLM, and other local model frameworks
"""
import logging
from typing import Optional, Dict, Any, List
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LocalModelInterface:
    """Base interface for local models"""
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        raise NotImplementedError


class HuggingFaceModel(LocalModelInterface):
    """Hugging Face Transformers model wrapper"""
    
    def __init__(self, model_path: str, device: str = "auto", load_in_8bit: bool = False, 
                 load_in_4bit: bool = False, **kwargs):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.kwargs = kwargs
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logging.info(f"Loading Hugging Face model from {model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": device if device != "auto" else "auto",
            }
            
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            model_kwargs.update(kwargs)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            logging.info(f"Successfully loaded model {model_path}")
            
        except ImportError as e:
            logging.error(f"Failed to import transformers or torch: {e}")
            logging.error("Please install with: pip install transformers torch")
            raise
        except Exception as e:
            logging.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 500, 
                 temperature: float = 0.0, **kwargs) -> str:
        """Generate response from messages"""
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.model.device.type != 'cpu':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.1,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                            skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string"""
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        # Fallback to simple concatenation
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system' or role == 'developer':
                prompt_parts.append(f"System: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant' or role == 'model':
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None and self.tokenizer is not None


class VLLMModel(LocalModelInterface):
    """vLLM model wrapper for fast inference"""
    
    def __init__(self, model_path: str = None, base_url: str = None, api_key: str = "EMPTY", **kwargs):
        self.model_path = model_path
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
        self.kwargs = kwargs
        
        if base_url:
            # Use OpenAI-compatible API endpoint
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
                logging.info(f"Connected to vLLM server at {base_url}")
            except ImportError:
                logging.error("openai package required for vLLM client mode")
                raise
        else:
            # Use vLLM directly
            try:
                from vllm import LLM
                logging.info(f"Loading vLLM model from {model_path}...")
                self.llm = LLM(model=model_path, **kwargs)
                logging.info(f"Successfully loaded vLLM model {model_path}")
            except ImportError:
                logging.error("vllm package required. Install with: pip install vllm")
                raise
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 500, 
                 temperature: float = 0.0, **kwargs) -> str:
        """Generate response from messages"""
        try:
            if self.client:
                # Use OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.model_path or "default",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
            else:
                # Use vLLM directly
                from vllm import SamplingParams
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(messages)
                
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.1,
                    **kwargs
                )
                
                outputs = self.llm.generate([prompt], sampling_params)
                return outputs[0].outputs[0].text.strip()
                
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system' or role == 'developer':
                prompt_parts.append(f"System: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant' or role == 'model':
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.client is not None or hasattr(self, 'llm')


class LocalModelManager:
    """Manages local model instances"""
    
    def __init__(self):
        self._models: Dict[str, LocalModelInterface] = {}
    
    def load_model(self, model_name: str, model_type: str = "huggingface", 
                   model_path: str = None, **kwargs) -> LocalModelInterface:
        """Load a local model"""
        
        if model_name in self._models:
            return self._models[model_name]
        
        if model_type == "huggingface":
            model = HuggingFaceModel(model_path or model_name, **kwargs)
        elif model_type == "vllm":
            model = VLLMModel(model_path or model_name, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self._models[model_name] = model
        return model
    
    def get_model(self, model_name: str) -> Optional[LocalModelInterface]:
        """Get a loaded model"""
        return self._models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded"""
        return model_name in self._models and self._models[model_name].is_available()


# Global instance
_model_manager: Optional[LocalModelManager] = None


def get_local_model_manager() -> LocalModelManager:
    """Get or create the global local model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalModelManager()
    return _model_manager
