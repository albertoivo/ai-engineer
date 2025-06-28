from openai import OpenAI

client = OpenAI()

def summarize_text(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input="Write a one-sentence bedtime story about a unicorn."
    )
    return response.output_text
