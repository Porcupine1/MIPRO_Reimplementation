import requests
from typing import Optional
from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS


class LMBackend:
    """interface to Ollama for LLM generation."""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """generate text from prompt using Ollama."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        response = requests.post(self.generate_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if "response" not in result:
            raise ValueError(f"Unexpected response format from Ollama API: {result}")
        return result["response"]
    
    def generate_query(self, question: str, instruction: str) -> str:
        """generate search query from question."""

        prompt = f"{instruction}\n\nQuestion: {question}\n\nSearch Query:"
        return self.generate(prompt).strip()
    
    def generate_answer(self, question: str, context: str, instruction: str) -> str:
        """generate answer from question and context."""

        prompt = f"{instruction}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(prompt).strip()