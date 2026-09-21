# tokentoken/router.py

def execute_compression(provider: str, model: str, system_prompt: str, user_text: str, api_key: str = None, host: str = None, base_url: str = None) -> str:
    if provider == "ollama":
        from ollama import Client
        client = Client(host=host or "http://localhost:11434")
        response = client.chat(model=model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ])
        return response['message']['content']

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        # Gemini 1.5 Pro is highlighted in the paper as the best compressor!
        response = client.models.generate_content(
            model=model,
            contents=[system_prompt + "\n\n" + user_text]
        )
        return response.text

    elif provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}]
        )
        return response.content[0].text
        
    raise ValueError(f"Unknown provider: {provider}")
