import os
from typing import List, Optional

import anthropic
from dotenv import load_dotenv

from core.core import Chunk, GenerativeModel

load_dotenv()

class AnthropicGenerativeModel(GenerativeModel):
    """Anthropic Claude implementation"""
    
    def __init__(self, model_name: str = "claude-opus-4-20250514", api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text based on prompt"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return message.content[0].text
    # def generate_with_streaming(self, prompt: str, max_tokens: int = 1000, **kwargs) :
    #     """Generate text with streaming response"""
        
    #     with self.client.stream(
    #         model=self.model_name,
    #         max_tokens=max_tokens,
    #         messages=[{"role": "user", "content": prompt}],
    #         **kwargs
    #     ) as stream:
    #         for text in stream.text_stream:
    #             print(text, end='', flush=True)
    #             yield text

    def generate_with_context(
        self, 
        query: str, 
        context_chunks: List[Chunk], 
        max_tokens: int = 1000, 
        **kwargs
    ) -> str:
        """Generate answer with context chunks"""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"Context {i}:\n{chunk.content}\n")
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""Based on the following context information, please answer the user's question.

                Context Information:
                {context_text}

                User Question: {query}

                Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""

        return self.generate(prompt, max_tokens, **kwargs)
