# from fastapi import FastAPI
# from pydantic import BaseModel
# from chat_bot_1 import run_chat
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Cấu hình CORS
# origins = [
#     "http://localhost:3000",   # React/Next.js dev server
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,          # hoặc ["*"] để cho phép tất cả
#     allow_credentials=True,
#     allow_methods=["*"],            # cho phép tất cả phương thức: GET, POST, PUT, DELETE...
#     allow_headers=["*"],            # cho phép tất cả header
# )
# class ChatRequest(BaseModel):
#     user_id: str
#     message: str

# class ChatResponse(BaseModel):
#     answer: str

# session = {"configurable": {"thread_id": }}

# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     answer = run_chat(request.message, session)
#     return {"answer": answer}

# @app.get("/")
# def root():
#     return {"msg": "API is running"}


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
