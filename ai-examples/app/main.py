from fastapi import FastAPI
from pydantic import BaseModel
from .openai_service import summarize_text

app = FastAPI()


class SummaryRequest(BaseModel):
    text: str

@app.post("/api/text-summarize")
def text_summarize(req: SummaryRequest):
    summary = summarize_text(req.text)
    return {"summary": summary}
