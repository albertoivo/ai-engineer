from openai import OpenAI
client = OpenAI()

try:
    response = client.responses.create(
        model="gpt-4.1",
        input="Write a one-sentence bedtime story about a unicorn."
    )
except Exception as e:
    print(f"An error occurred: {e}")

print(response.output_text)