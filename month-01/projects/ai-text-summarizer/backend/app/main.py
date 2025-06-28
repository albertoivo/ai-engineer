from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .openai_service import summarize_text

app = FastAPI()

# Permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # porta padrão do Vite
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    text: str

@app.post("/api/summarize")
def summarize(req: SummaryRequest):
    summary = summarize_text(req.text)
    return {"summary": summary}
