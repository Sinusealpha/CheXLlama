# llm/client.py
import os
from groq import Groq
import config

def get_llm_response(prompt, model=config.LLM_MODEL_NAME, temp=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS):
    """
    Sends a single prompt to the Groq API and returns the response.
    This function is designed for a one-shot request, not a continuous chat.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")
        
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temp,
            max_tokens=max_tokens
        )
        
        bot_response = response.choices[0].message.content
        return bot_response
        
    except Exception as e:
        print(f"An error occurred while communicating with the LLM API: {e}")
        return None
