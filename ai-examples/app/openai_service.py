from openai import OpenAI

client = OpenAI()

def summarize_text(text: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    summary = response.choices[0].message.content.strip()
    return summary
