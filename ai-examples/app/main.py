from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from .openai_service import summarize_text, generate_image_description

app = FastAPI()


class SummaryRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

@app.post("/api/text-summarize")
def text_summarize(req: SummaryRequest):
    response = summarize_text(req.text)
    return {"response": response}

@app.post("/api/image")
def generate_image_description_route(req: SummaryRequest):
    response = generate_image_description(req.url)
    return {"response": response}
