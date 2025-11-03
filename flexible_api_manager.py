"""
Flexible API Manager for Oracle Benchmark
Provides on-demand API client initialization to avoid requiring all API keys
"""
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FlexibleAPIManager:
    """
    Manages API clients for different model families.
    Only initializes clients when needed and when API keys are available.
    """
    
    def __init__(self):
        self._clients: Dict[str, Any] = {}
        self._api_keys: Dict[str, Optional[str]] = {
            'openai': os.getenv("OPENAI_API_KEY"),
            'anthropic': os.getenv("ANTHROPIC_API_KEY"),
            'gemini': os.getenv("GEMINI_API_KEY"),
            'alibaba': os.getenv("ALIBABA_API_KEY"),
            'deepseek': os.getenv("DEEPSEEK_API_KEY"),
            'openrouter': os.getenv("OPENROUTER_API_KEY"),
        }
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for a provider"""
        key = self._api_keys.get(provider)
        return key is not None and key != "" and key != "xxx"
    
    def get_openai_client(self):
        """Get or create OpenAI client"""
        if 'openai' not in self._clients:
            if not self.has_api_key('openai'):
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
            
            from openai import OpenAI
            self._clients['openai'] = OpenAI(api_key=self._api_keys['openai'])
            logging.info("OpenAI client initialized")
        
        return self._clients['openai']
    
    def get_claude_client(self):
        """Get or create Claude client"""
        if 'claude' not in self._clients:
            if not self.has_api_key('anthropic'):
                raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env file")
            
            from anthropic import Anthropic
            self._clients['claude'] = Anthropic(api_key=self._api_keys['anthropic'])
            logging.info("Claude client initialized")
        
        return self._clients['claude']
    
    def get_gemini_client(self):
        """Get or create Gemini client"""
        if 'gemini' not in self._clients:
            if not self.has_api_key('gemini'):
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env file")
            
            from google import genai
            self._clients['gemini'] = genai.Client(api_key=self._api_keys['gemini'])
            logging.info("Gemini client initialized")
        
        return self._clients['gemini']
    
    def get_qwen_client(self):
        """Get or create Qwen client"""
        if 'qwen' not in self._clients:
            if not self.has_api_key('alibaba'):
                raise ValueError("Alibaba API key not found. Please set ALIBABA_API_KEY in .env file")
            
            from openai import OpenAI
            self._clients['qwen'] = OpenAI(
                api_key=self._api_keys['alibaba'],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            logging.info("Qwen client initialized")
        
        return self._clients['qwen']
    
    def get_deepseek_client(self):
        """Get or create DeepSeek client"""
        if 'deepseek' not in self._clients:
            if not self.has_api_key('deepseek'):
                raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY in .env file")
            
            from openai import OpenAI
            self._clients['deepseek'] = OpenAI(
                api_key=self._api_keys['deepseek'],
                base_url="https://api.deepseek.com"
            )
            logging.info("DeepSeek client initialized")
        
        return self._clients['deepseek']
    
    def get_openrouter_headers(self):
        """Get OpenRouter headers"""
        if not self.has_api_key('openrouter'):
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
        
        return {
            "Authorization": f"Bearer {self._api_keys['openrouter']}",
            "Content-Type": "application/json"
        }
    
    def get_client_for_family(self, model_family: str):
        """Get appropriate client for a model family"""
        if model_family == 'gpt':
            return self.get_openai_client()
        elif model_family == 'claude':
            return self.get_claude_client()
        elif model_family == 'gemini':
            return self.get_gemini_client()
        elif model_family == 'qwen':
            return self.get_qwen_client()
        elif model_family == 'deepseek':
            return self.get_deepseek_client()
        elif model_family == 'llama':
            # Llama uses OpenRouter, return headers
            return self.get_openrouter_headers()
        elif model_family == 'local':
            # Local models don't need API clients
            return None
        else:
            raise ValueError(f"Unknown model family: {model_family}")
    
    def check_all_available_keys(self) -> Dict[str, bool]:
        """Check which API keys are available"""
        return {
            provider: self.has_api_key(provider)
            for provider in self._api_keys.keys()
        }
    
    def get_available_model_families(self) -> list:
        """Get list of model families with available API keys"""
        available = []
        
        if self.has_api_key('openai'):
            available.append('gpt')
        if self.has_api_key('anthropic'):
            available.append('claude')
        if self.has_api_key('gemini'):
            available.append('gemini')
        if self.has_api_key('alibaba'):
            available.append('qwen')
        if self.has_api_key('deepseek'):
            available.append('deepseek')
        if self.has_api_key('openrouter'):
            available.append('llama')
        
        # Local models are always available (if dependencies are installed)
        available.append('local')
        
        return available


# Global instance
_api_manager: Optional[FlexibleAPIManager] = None


def get_api_manager() -> FlexibleAPIManager:
    """Get or create the global API manager instance"""
    global _api_manager
    if _api_manager is None:
        _api_manager = FlexibleAPIManager()
    return _api_manager
