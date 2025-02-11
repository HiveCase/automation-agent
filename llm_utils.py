import os
import requests

API_PROXY_URL = "https://api.aiproxy.com/v1/generation/generate"

def call_llm(prompt):
    """
    Calls the LLM (GPT-4o-Mini) with the given prompt.
    """
    api_token = os.environ.get("AIPROXY_TOKEN")
    if not api_token:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "GPT-4o-Mini",
        "prompt": prompt,
        "max_tokens": 200  # Keep prompts short to respect the 20-second limit
    }
    try:
        response = requests.post(API_PROXY_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API call failed: {e}")
