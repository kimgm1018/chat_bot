import os
from dotenv import load_dotenv

# ✅ 이 코드가 다른 import나 FastAPI 앱 생성보다 먼저 실행되어야 합니다.
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from chat_bot_1 import run_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

# ✅ 유저별 세션 저장소 (딕셔너리)
sessions: dict[str, dict] = {}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 유저별 세션이 없으면 새로 만듦
    if request.user_id not in sessions:
        sessions[request.user_id] = {
            "configurable": {
                "thread_id": request.user_id, 
                "user_id": request.user_id
            }
        }

    # 해당 유저 세션 불러오기
    session = sessions[request.user_id]

    # run_chat 실행
    answer = run_chat(request.message, session)
    return {"answer": answer}

@app.get("/")
def root():
    return {"msg": "API is running"}
