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

def generate_text(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    generated_text = response.choices[0].message.content.strip()
    return generated_text

def chat_with_model(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that engages in conversation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    chat_response = response.choices[0].message.content.strip()
    return chat_response

def translate_text(text: str, target_language: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Please translate the following text to {target_language}:\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    translated_text = response.choices[0].message.content.strip()
    return translated_text

def answer_question(question: str, context: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Based on the following context, please answer the question:\n\nContext: {context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def generate_code(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    generated_code = response.choices[0].message.content.strip()
    return generated_code

def analyze_sentiment(text: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Please analyze the sentiment of the following text:\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes sentiment."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    sentiment_analysis = response.choices[0].message.content.strip()
    return sentiment_analysis

def extract_keywords(text: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Please extract keywords from the following text:\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts keywords."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    keywords = response.choices[0].message.content.strip()
    return keywords

def generate_image_description(image_url: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    prompt = f"Please describe the image at the following URL:\n\n{image_url}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes images."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    image_description = response.choices[0].message.content.strip()
    return image_description

def generate_poem(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates poems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    poem = response.choices[0].message.content.strip()
    return poem

def generate_story(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates stories."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    story = response.choices[0].message.content.strip()
    return story

def generate_joke(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates jokes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    joke = response.choices[0].message.content.strip()
    return joke

def generate_recipe(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates recipes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    recipe = response.choices[0].message.content.strip()
    return recipe

def generate_business_idea(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates business ideas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    business_idea = response.choices[0].message.content.strip()
    return business_idea

def generate_marketing_slogan(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates marketing slogans."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    slogan = response.choices[0].message.content.strip()
    return slogan

def generate_social_media_post(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates social media posts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    social_media_post = response.choices[0].message.content.strip()
    return social_media_post

def generate_email(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates emails."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    email_content = response.choices[0].message.content.strip()
    return email_content

def generate_blog_post(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates blog posts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    blog_post = response.choices[0].message.content.strip()
    return blog_post

def generate_advertisement(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates advertisements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    advertisement = response.choices[0].message.content.strip()
    return advertisement

def generate_product_description(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates product descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    product_description = response.choices[0].message.content.strip()
    return product_description

def generate_faq(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates FAQs."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    faq = response.choices[0].message.content.strip()
    return faq

def generate_tweet(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates tweets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    tweet = response.choices[0].message.content.strip()
    return tweet

def generate_news_article(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates news articles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    news_article = response.choices[0].message.content.strip()
    return news_article

def generate_script(prompt: str, model: str = "gpt-4", max_tokens: int = 150) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates scripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    script = response.choices[0].message.content.strip()
    return script