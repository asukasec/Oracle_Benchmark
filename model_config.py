"""
Model Configuration Manager for Oracle Benchmark
Provides YAML-based model configuration and management
"""
import os
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelConfig:
    """Configuration for a single model"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict.get('name')
        self.family = config_dict.get('family')
        self.type = config_dict.get('type', 'api')  # api, local, vllm
        self.api_name = config_dict.get('api_name', self.name)
        self.path = config_dict.get('path')
        self.base_url = config_dict.get('base_url')
        self.requires_api_key = config_dict.get('requires_api_key', True)
        self.supports_thinking = config_dict.get('supports_thinking', False)
        self.device = config_dict.get('device', 'auto')
        self.load_in_8bit = config_dict.get('load_in_8bit', False)
        self.load_in_4bit = config_dict.get('load_in_4bit', False)
        self.extra_params = config_dict.get('extra_params', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'family': self.family,
            'type': self.type,
            'api_name': self.api_name,
            'path': self.path,
            'base_url': self.base_url,
            'requires_api_key': self.requires_api_key,
            'supports_thinking': self.supports_thinking,
            'device': self.device,
            'load_in_8bit': self.load_in_8bit,
            'load_in_4bit': self.load_in_4bit,
            'extra_params': self.extra_params,
        }


class ModelConfigManager:
    """Manages model configurations from YAML file"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.models: Dict[str, ModelConfig] = {}
        self._load_default_models()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
    
    def _load_default_models(self):
        """Load default model configurations"""
        # Default API models
        default_configs = [
            # GPT models
            {'name': 'gpt-4o', 'family': 'gpt', 'type': 'api', 'api_name': 'gpt-4o-2024-08-06'},
            {'name': 'gpt-4o-mini', 'family': 'gpt', 'type': 'api', 'api_name': 'gpt-4o-mini-2024-07-18'},
            {'name': 'gpt-4.1', 'family': 'gpt', 'type': 'api', 'api_name': 'gpt-4.1-2025-04-14'},
            {'name': 'o1', 'family': 'gpt', 'type': 'api', 'api_name': 'o1-2024-12-17'},
            {'name': 'o3-mini', 'family': 'gpt', 'type': 'api', 'api_name': 'o3-mini-2025-01-31'},
            {'name': 'o4-mini', 'family': 'gpt', 'type': 'api', 'api_name': 'o4-mini-2025-04-16'},
            
            # Claude models
            {'name': 'claude-3.5-sonnet', 'family': 'claude', 'type': 'api', 'api_name': 'claude-3-5-sonnet-20241022'},
            {'name': 'claude-3.7-sonnet', 'family': 'claude', 'type': 'api', 'api_name': 'claude-3-7-sonnet-20250219', 'supports_thinking': True},
            {'name': 'claude-4-sonnet', 'family': 'claude', 'type': 'api', 'api_name': 'claude-sonnet-4-20250514', 'supports_thinking': True},
            
            # Gemini models
            {'name': 'gemini-2.0-flash', 'family': 'gemini', 'type': 'api', 'api_name': 'gemini-2.0-flash'},
            {'name': 'gemini-2.5-flash', 'family': 'gemini', 'type': 'api', 'api_name': 'gemini-2.5-flash', 'supports_thinking': True},
            {'name': 'gemini-2.5-pro', 'family': 'gemini', 'type': 'api', 'api_name': 'gemini-2.5-pro', 'supports_thinking': True},
            
            # Qwen models
            {'name': 'qwen-plus', 'family': 'qwen', 'type': 'api', 'api_name': 'qwen-plus-latest', 'supports_thinking': True},
            {'name': 'qwen3-32b', 'family': 'qwen', 'type': 'api', 'api_name': 'qwen3-32b', 'supports_thinking': True},
            {'name': 'qwq-plus', 'family': 'qwen', 'type': 'api', 'api_name': 'qwq-plus', 'supports_thinking': True},
            
            # DeepSeek models
            {'name': 'deepseek-v3', 'family': 'deepseek', 'type': 'api', 'api_name': 'deepseek-chat'},
            {'name': 'deepseek-r1', 'family': 'deepseek', 'type': 'api', 'api_name': 'deepseek-reasoner'},
            
            # Llama models (via OpenRouter)
            {'name': 'llama-4-scout', 'family': 'llama', 'type': 'api', 'api_name': 'meta-llama/llama-4-scout'},
            {'name': 'llama-4-marverick', 'family': 'llama', 'type': 'api', 'api_name': 'meta-llama/llama-4-maverick'},
        ]
        
        for config_dict in default_configs:
            model_config = ModelConfig(config_dict)
            self.models[model_config.name] = model_config
    
    def _load_config_file(self, config_path: str):
        """Load model configurations from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return
            
            models_data = config_data.get('models', [])
            for model_dict in models_data:
                model_config = ModelConfig(model_dict)
                self.models[model_config.name] = model_config
                logging.info(f"Loaded model configuration: {model_config.name}")
                
        except Exception as e:
            logging.error(f"Error loading config file {config_path}: {e}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model"""
        return self.models.get(model_name)
    
    def get_models_by_family(self, family: str) -> List[ModelConfig]:
        """Get all models in a family"""
        return [m for m in self.models.values() if m.family == family]
    
    def get_available_models(self, api_manager) -> List[ModelConfig]:
        """Get models that can be used with available API keys"""
        available = []
        
        for model in self.models.values():
            if model.type == 'local':
                # Local models don't need API keys
                available.append(model)
            elif model.type == 'api':
                # Check if required API key is available
                if model.family == 'gpt' and api_manager.has_api_key('openai'):
                    available.append(model)
                elif model.family == 'claude' and api_manager.has_api_key('anthropic'):
                    available.append(model)
                elif model.family == 'gemini' and api_manager.has_api_key('gemini'):
                    available.append(model)
                elif model.family == 'qwen' and api_manager.has_api_key('alibaba'):
                    available.append(model)
                elif model.family == 'deepseek' and api_manager.has_api_key('deepseek'):
                    available.append(model)
                elif model.family == 'llama' and api_manager.has_api_key('openrouter'):
                    available.append(model)
        
        return available
    
    def add_model(self, model_config: ModelConfig):
        """Add a model configuration"""
        self.models[model_config.name] = model_config
    
    def save_config(self, output_path: str):
        """Save current configurations to YAML file"""
        config_data = {
            'models': [m.to_dict() for m in self.models.values()]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logging.info(f"Saved model configurations to {output_path}")
    
    def list_all_models(self) -> List[str]:
        """Get list of all model names"""
        return list(self.models.keys())
    
    def list_model_families(self) -> List[str]:
        """Get list of all model families"""
        families = set(m.family for m in self.models.values())
        return sorted(list(families))


# Global instance
_config_manager: Optional[ModelConfigManager] = None


def get_model_config_manager(config_path: Optional[str] = None) -> ModelConfigManager:
    """Get or create the global model config manager instance"""
    global _config_manager
    if _config_manager is None or config_path:
        _config_manager = ModelConfigManager(config_path)
    return _config_manager
