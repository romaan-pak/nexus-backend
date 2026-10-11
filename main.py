from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


class ChatIn(BaseModel):
    message: str
    history: list = []


@app.get("/")
def root():
    return {"status": "ok", "ai_ready": bool(GROQ_KEY)}


@app.post("/api/chat")
def chat(payload: ChatIn):
    if not GROQ_KEY:
        return {"reply": "No API key configured on the server."}

    messages = [{"role": "system", "content": "You are Nexus AI, a helpful assistant."}]
    messages += payload.history
    messages.append({"role": "user", "content": payload.message})

    res = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "max_tokens": 800},
        timeout=60,
    )

    if res.status_code != 200:
        return {"reply": f"Error: {res.text[:200]}"}

    return {"reply": res.json()["choices"][0]["message"]["content"]}
