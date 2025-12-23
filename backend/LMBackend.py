import requests
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS, MAX_PARALLEL_WORKERS


class LMBackend:
    """interface to Ollama for LLM generation with parallel processing support."""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_workers: int = MAX_PARALLEL_WORKERS
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.generate_url = f"{base_url}/api/generate"
        self.max_workers = max_workers  # Number of parallel requests
    
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
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to generate from
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            List of generated texts in the same order as prompts
        """
        def _generate_single(prompt: str) -> str:
            return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(_generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results in order
            results = [None] * len(prompts)
            for future in as_completed(future_to_prompt):
                idx = future_to_prompt[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Error generating for prompt {idx}: {e}")
        
        return results
    
    def generate_query(self, question: str, instruction: str) -> str:
        """generate search query from question."""

        prompt = f"{instruction}\n\nQuestion: {question}\n\nSearch Query:"
        return self.generate(prompt).strip()
    
    def generate_answer(self, question: str, context: str, instruction: str) -> str:
        """generate answer from question and context."""

        prompt = f"{instruction}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(prompt).strip()