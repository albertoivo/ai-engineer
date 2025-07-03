import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .openai_service import (
    analyze_sentiment,
    answer_question,
    chat_with_model,
    extract_keywords,
    generate_advertisement,
    generate_blog_post,
    generate_business_idea,
    generate_code,
    generate_email,
    generate_faq,
    generate_image_description,
    generate_joke,
    generate_marketing_slogan,
    generate_news_article,
    generate_poem,
    generate_product_description,
    generate_recipe,
    generate_script,
    generate_social_media_post,
    generate_story,
    generate_text,
    generate_tweet,
    summarize_text,
    summarize_youtube_video,
    translate_text,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenAI API Wrapper",
    description="API para interagir com o serviço OpenAI",
    version="1.0.0",
)


# Modelos de requisição
class TextRequest(BaseModel):
    text: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 150


class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 150


class TranslationRequest(BaseModel):
    text: str
    target_language: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 150


class QuestionRequest(BaseModel):
    question: str
    context: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 150


class ImageRequest(BaseModel):
    image_url: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 150


class YouTubeRequest(BaseModel):
    video_url: str
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 500


# Middleware para logar requisições
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request path: {request.url.path}")
    response = await call_next(request)
    return response


# Rotas
@app.post("/api/text-summarize")
async def text_summarize(req: TextRequest):
    try:
        logger.info(f"Summarizing text: {req.text[:50]}...")
        response = summarize_text(req.text, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in text_summarize: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-text")
async def api_generate_text(req: PromptRequest):
    try:
        logger.info(f"Generating text for prompt: {req.prompt[:50]}...")
        response = generate_text(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(req: PromptRequest):
    try:
        logger.info(f"Chatting with prompt: {req.prompt[:50]}...")
        response = chat_with_model(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def api_translate(req: TranslationRequest):
    try:
        logger.info(f"Translating text to {req.target_language}: {req.text[:50]}...")
        response = translate_text(
            req.text, req.target_language, req.model, req.max_tokens
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in translate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/answer-question")
async def api_answer_question(req: QuestionRequest):
    try:
        logger.info(f"Answering question: {req.question[:50]}...")
        response = answer_question(req.question, req.context, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-code")
async def api_generate_code(req: PromptRequest):
    try:
        logger.info(f"Generating code for prompt: {req.prompt[:50]}...")
        response = generate_code(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-sentiment")
async def api_analyze_sentiment(req: TextRequest):
    try:
        logger.info(f"Analyzing sentiment: {req.text[:50]}...")
        response = analyze_sentiment(req.text, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-keywords")
async def api_extract_keywords(req: TextRequest):
    try:
        logger.info(f"Extracting keywords from: {req.text[:50]}...")
        response = extract_keywords(req.text, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in extract_keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/describe-image")
async def api_describe_image(req: ImageRequest):
    try:
        logger.info(f"Describing image at: {req.image_url}")
        response = generate_image_description(req.image_url, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in describe_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-poem")
async def api_generate_poem(req: PromptRequest):
    try:
        logger.info(f"Generating poem for prompt: {req.prompt[:50]}...")
        response = generate_poem(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_poem: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-story")
async def api_generate_story(req: PromptRequest):
    try:
        logger.info(f"Generating story for prompt: {req.prompt[:50]}...")
        response = generate_story(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_story: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-joke")
async def api_generate_joke(req: PromptRequest):
    try:
        logger.info(f"Generating joke for prompt: {req.prompt[:50]}...")
        response = generate_joke(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_joke: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-recipe")
async def api_generate_recipe(req: PromptRequest):
    try:
        logger.info(f"Generating recipe for prompt: {req.prompt[:50]}...")
        response = generate_recipe(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-business-idea")
async def api_generate_business_idea(req: PromptRequest):
    try:
        logger.info(f"Generating business idea for prompt: {req.prompt[:50]}...")
        response = generate_business_idea(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_business_idea: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-marketing-slogan")
async def api_generate_slogan(req: PromptRequest):
    try:
        logger.info(f"Generating slogan for prompt: {req.prompt[:50]}...")
        response = generate_marketing_slogan(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_slogan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-social-post")
async def api_generate_social_post(req: PromptRequest):
    try:
        logger.info(f"Generating social media post for prompt: {req.prompt[:50]}...")
        response = generate_social_media_post(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_social_post: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-email")
async def api_generate_email(req: PromptRequest):
    try:
        logger.info(f"Generating email for prompt: {req.prompt[:50]}...")
        response = generate_email(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-blog-post")
async def api_generate_blog_post(req: PromptRequest):
    try:
        logger.info(f"Generating blog post for prompt: {req.prompt[:50]}...")
        response = generate_blog_post(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_blog_post: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-ad")
async def api_generate_ad(req: PromptRequest):
    try:
        logger.info(f"Generating advertisement for prompt: {req.prompt[:50]}...")
        response = generate_advertisement(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_ad: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-product-description")
async def api_generate_product_description(req: PromptRequest):
    try:
        logger.info(f"Generating product description for prompt: {req.prompt[:50]}...")
        response = generate_product_description(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_product_description: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-faq")
async def api_generate_faq(req: PromptRequest):
    try:
        logger.info(f"Generating FAQ for prompt: {req.prompt[:50]}...")
        response = generate_faq(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_faq: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-tweet")
async def api_generate_tweet(req: PromptRequest):
    try:
        logger.info(f"Generating tweet for prompt: {req.prompt[:50]}...")
        response = generate_tweet(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_tweet: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-news-article")
async def api_generate_news_article(req: PromptRequest):
    try:
        logger.info(f"Generating news article for prompt: {req.prompt[:50]}...")
        response = generate_news_article(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_news_article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-script")
async def api_generate_script(req: PromptRequest):
    try:
        logger.info(f"Generating script for prompt: {req.prompt[:50]}...")
        response = generate_script(req.prompt, req.model, req.max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in generate_script: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/youtube-summary")
async def api_youtube_summary(req: YouTubeRequest):
    """
    Gera resumo estruturado de vídeo do YouTube
    """
    try:
        logger.info(f"Summarizing YouTube video: {req.video_url}")
        response = summarize_youtube_video(
            video_url=req.video_url, model=req.model, max_tokens=req.max_tokens
        )
        return {"success": True, "video_url": req.video_url, "summary": response}
    except ValueError as e:
        logger.error(f"Validation error in YouTube summary: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in YouTube summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Rota de informações sobre a API
@app.get("/")
async def root():
    return {
        "app": "OpenAI API Wrapper",
        "version": "1.0.0",
        "endpoints": {
            "text_summarize": "/api/text-summarize",
            "generate_text": "/api/generate-text",
            "chat": "/api/chat",
            "translate": "/api/translate",
            "answer_question": "/api/answer-question",
            "generate_code": "/api/generate-code",
            "analyze_sentiment": "/api/analyze-sentiment",
            "extract_keywords": "/api/extract-keywords",
            "describe_image": "/api/describe-image",
            "generate_poem": "/api/generate-poem",
            "generate_story": "/api/generate-story",
            "generate_joke": "/api/generate-joke",
            "generate_recipe": "/api/generate-recipe",
            "generate_business_idea": "/api/generate-business-idea",
            "generate_marketing_slogan": "/api/generate-marketing-slogan",
            "generate_social_post": "/api/generate-social-post",
            "generate_email": "/api/generate-email",
            "generate_blog_post": "/api/generate-blog-post",
            "generate_ad": "/api/generate-ad",
            "generate_product_description": "/api/generate-product-description",
            "generate_faq": "/api/generate-faq",
            "generate_tweet": "/api/generate-tweet",
            "generate_news_article": "/api/generate-news-article",
            "generate_script": "/api/generate-script",
            "youtube_summary": "/api/youtube-summary",
        },
        "documentation": "/docs",
    }
