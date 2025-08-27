## Import Line ------------------------------------------------------

import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

# TypeAnotation
from typing_extensions import TypedDict, Literal, List, Optional, Any, Dict
from typing_extensions import Sequence, Annotated

# Graph
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Langchain
from langchain_openai import ChatOpenAI

# Langchain_Core
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langchain rag
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# DB
from datetime import datetime
from elasticsearch import Elasticsearch
from pytz import timezone
import time


class State2(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  log_in_id: str|None
  reserv_company: Annotated[Optional[str], lambda x, y: y or x]
  reserv_name: Annotated[Optional[str], lambda x, y: y or x]
  contact_email: Annotated[Optional[str], lambda x, y: y or x]
  contact_phonenum: Annotated[Optional[str], lambda x, y: y or x]
  reserv_purpose: Annotated[Optional[str], lambda x, y: y or x]

#____________________________________ DB ____________________________________

def sanitize_id(log_in_id: str) -> str:
    if not log_in_id:
        return "default"
    safe = re.sub(r'[^a-z0-9_\-]', '', log_in_id.lower())
    if not safe:
        safe = "default"
    return safe

def create_user_table_if_not_exists(log_in_id: str):
    es = Elasticsearch("http://localhost:9200")
    
    sanitized_id = sanitize_id(log_in_id)
    user_chat_index = f"chat_{sanitized_id}"
    reservation_index = "reservation_memory"

    if not es.indices.exists(index=user_chat_index):
        es.indices.create(
            index=user_chat_index,
            body={
                "mappings": {
                    "properties": {
                        "log_in_id": {"type": "keyword"},
                        "role": {"type": "keyword"},
                        "content": {"type": "text"},
                        "time_stamp": {"type": "date"}
                    }
                }
            }
        )

    if not es.indices.exists(index=reservation_index):
        es.indices.create(
            index=reservation_index,
            body={
                "mappings": {
                    "properties": {
                        "log_in_id": {"type": "keyword"},
                        "reserv_company": {"type": "keyword"},
                        "reserv_name": {"type": "keyword"},
                        "contact_email": {"type": "keyword"},
                        "contact_phonenum": {"type": "keyword"},
                        "reserv_purpose": {"type": "text"},
                        "time_stamp": {"type" : "date"}
                    }
                }
            }
        )
                  
def save_message(log_in_id: str, role: str, content: str):
    es = Elasticsearch("http://localhost:9200")
    sanitized_id = sanitize_id(log_in_id)

    create_user_table_if_not_exists(log_in_id)

    document = {
        "log_in_id": sanitized_id,
        "role": role,
        "content": content,
        "time_stamp": datetime.now(timezone('UTC')).isoformat() # ✅ pytz 사용
    }
    
    es.index(
        index=f"chat_{sanitized_id}",
        document=document
    )

def load_chat_history(log_in_id: str) -> List:
    es = Elasticsearch("http://localhost:9200")
    sanitized_id = sanitize_id(log_in_id)
    
    create_user_table_if_not_exists(log_in_id)

    query = {
        "size": 100,
        "query": {
            "match": {"log_in_id": sanitized_id}
        },
        "sort": [
            {"time_stamp": {"order": "asc"}}
        ]
    }
    
    res = es.search(index=f"chat_{sanitized_id}", body=query)
    
    messages = []
    for hit in res["hits"]["hits"]:
        source = hit["_source"]
        if source["role"] == "human":
            messages.append(HumanMessage(content=source["content"]))
        elif source["role"] == "ai":
            messages.append(AIMessage(content=source["content"]))
        elif source["role"] == "tool":
            messages.append(ToolMessage(content=source["content"], tool_call_id="dummy"))
            
    return messages

def save_reserv(log_in_id: str, reserv_company: str, reserv_name: str, contact_email: str, contact_phonenum: str, reserv_purpose: str):
    es = Elasticsearch("http://localhost:9200")
    sanitized_id = sanitize_id(log_in_id)
    create_user_table_if_not_exists(log_in_id)

    reservation = {"log_in_id": sanitized_id,
                    "reserv_company": reserv_company,
                    "reserv_name": reserv_name,
                    "contact_email": contact_email,
                    "contact_phonenum": contact_phonenum,
                    "reserv_purpose": reserv_purpose,
                    "time_stamp": datetime.now(timezone('UTC')).isoformat() # ✅ pytz 사용
                   }
    es.index(index="reservation_memory", document=reservation)
    
def load_latest_reservation(log_in_id: str) -> Optional[Dict[str, str]]:
    es = Elasticsearch("http://localhost:9200")
    sanitized_id = sanitize_id(log_in_id)
    create_user_table_if_not_exists(log_in_id)

    query = {
        "size": 1,
        "query": {
            "match": {"log_in_id": sanitized_id}
        },
        "sort": [
            {"time_stamp": {"order": "desc"}}
        ]
    }

    res = es.search(index="reservation_memory", body=query)

    hits = res["hits"]["hits"]
    if hits:
        return hits[0]["_source"]
    return None

#____________________________________ DB ____________________________________


# ---------------------------- Tool ----------------------------
# @tool
# def find_company_info(query: str) -> str:
#     """SPMED(회사) 관련 문서를 검색해 상위 결과를 문자열로 반환합니다."""
#     print("1111")
#     embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
#     db = FAISS.load_local(os.getenv("FAISS_INDEX_PATH"), embedding, allow_dangerous_deserialization=True)
#     docs = db.as_retriever(k=3).invoke(query)
    
#     if not docs:
#         return ""
#     return "Top matches:\n" + "\n".join(f"- {d.page_content}" for d in docs)
# # elastic search -> string -> type : text??

# chat_bot_2.py 파일 내 find_company_info 함수 수정

@tool
def find_company_info(query: str) -> str:
    """SPMED(회사) 관련 문서를 검색해 상위 결과를 문자열로 반환합니다."""
    
    # Elasticsearch 클라이언트 초기화 (필요시 전역 변수로 관리)
    es = Elasticsearch("http://localhost:9200")
    
    # 1. OpenAI Embeddings 모델을 사용해 사용자 쿼리를 벡터로 변환합니다.
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    query_vector = embeddings_model.embed_query(query)

    # 2. Elasticsearch에 벡터 검색 쿼리를 보냅니다.
    search_query = {
        "size": 10, # 상위 3개 문서 검색
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    # 3. 검색을 실행하고 결과를 가져옵니다.
    # start = time.time()
    try:
        response = es.search(index="spmed_info", body=search_query)
        hits = response.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"Elasticsearch 검색 중 오류 발생: {e}")
        return ""

    # 4. 검색 결과에서 문서 내용을 추출합니다.
    if not hits:
        return "관련 문서를 찾을 수 없습니다."
        
    docs = []
    for hit in hits:
        source = hit.get("_source", {})
        if source.get("content"):
            docs.append(source["content"])
        elif source.get("answer"):
            # FAQ 데이터의 경우 "answer" 필드를 사용
            docs.append(f"Q: {source['question']}\nA: {source['answer']}")

    # end = time.time()

    # es_time = end - start
    # print(es_time)

    # embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    # start = time.time()
    # db = FAISS.load_local(os.getenv("FAISS_INDEX_PATH"), embedding, allow_dangerous_deserialization=True)
    # docs2 = db.as_retriever(k=10).invoke(query)
    # end = time.time()

    # faiss_time = end - start
    # print(faiss_time)

    
    # if not docs:
    #     return ""
    # return "Top matches:\n" + "\n".join(f"- {d.page_content}" for d in docs)
    # elastic search -> string -> type : text??

    
    return "Top matches:\n" + "\n".join(f"- {d}" for d in docs)

@tool
def make_reservation(
    reserv_company: str,
    reserv_name: str,
    contact_email: str,
    contact_phonenum: str,
    reserv_purpose: str
) -> Dict[str, str]:
  """
  이용자가 회사의 서비스(검사, 상담, 의뢰)를 챗봇을 통해 예약하고 싶을 때 예약을 진행하는 tool입니다.
  필요한 정보로는 회사(의뢰 기관)의 이름, 책임자의 이름, 이메일, 연락 가능한 전화번호, 서비스 이용 목적이 필수적입니다.
  """
  output = {"reserv_company" : reserv_company,
            "reserv_name" : reserv_name,
            "contact_email" : contact_email,
            "contact_phonenum" : contact_phonenum,
            "reserv_purpose" : reserv_purpose}
  return output

@tool
def send_email(reserv_company: str,
               reserv_name: str,
               contact_email: str,
               contact_phonenum: str,
               reserv_purpose: str) -> dict:
  """
  의뢰에 필요한 모든 필드가 체워지고, 의뢰 정보 확인을 받은 뒤, 의뢰 정보를 메일로 보내는 tool입니다.
  만약 모든 필드가 체워지지 않았다면, 메일을 보내지 않고 부족한 정보에 대한 정보를 반환합니다.
  """
  essential_info = [reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose]
  lack_info = {}
  gained_info = {}

  for i in essential_info:
      if i == None or i ==  '':
          lack_info[i] = i
      else:
          gained_info[i] = i
  
  output = {"gained_info" : gained_info,
            "lack_info" : lack_info}
  
  if len(output['lack_info']) > 0:
      return output

  import smtplib
  from email.mime.text import MIMEText

  # SMTP 서버 정보
  SMTP_HOST = "email-smtp.ap-northeast-2.amazonaws.com"
  SMTP_PORT = 587

  SMTP_USER = os.getenv("SMTP_USER") # AWS SES에서 발급
  SMTP_PASS = os.getenv("SMTP_PASS") # AWS SES에서 발급

  # 메일 내용 구성
  sender = "help@spmed.kr"
  recipient = contact_email
  subject = "SPMED 의뢰 정보 확인"
  body =f"""<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="UTF-8">
    <title>의뢰 확인 안내</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
  </head>
  <body style="margin:0; padding:0; background:#f4f6f8;">
    <!-- Wrapper -->
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#f4f6f8;">
      <tr>
        <td align="center" style="padding:24px 12px;">
          <!-- Container -->
          <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:600px; background:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <!-- Header -->
            <tr>
              <td align="center" style="background:#22c55e; padding:20px 16px;">
                <h1 style="margin:0; font-size:20px; line-height:28px; color:#ffffff; font-family:Arial,Helvetica,sans-serif;">
                  의뢰 정보가 전달되었습니다
                </h1>
              </td>
            </tr>

            <!-- Greeting -->
            <tr>
              <td style="padding:24px 24px 8px; font-family:Arial,Helvetica,sans-serif; color:#111827;">
                <p style="margin:0; font-size:16px; line-height:24px;">
                  <strong>{reserv_name}</strong> 님, 의뢰 정보가 정상적으로 전달되었습니다.
                </p>
                <p style="margin:8px 0 0; font-size:14px; line-height:22px; color:#4b5563;">
                  담당 부서에서 최대한 빠르게 연락 드리겠습니다.
                </p>
                <p style="margin:4px 0 0; font-size:14px; line-height:22px; color:#4b5563;">
                  아래에서 의뢰 정보를 확인해주세요.
                </p>
              </td>
            </tr>

            <!-- Details Card -->
            <tr>
              <td style="padding:12px 24px 8px;">
                <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="border:1px solid #e5e7eb; border-radius:10px;">
                  <tr>
                    <td style="padding:16px 18px; font-family:Arial,Helvetica,sans-serif;">
                      <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280; width:120px;"> 의뢰사/기관</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{reserv_company}</strong></td>
                        </tr>
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280; width:120px;"> 의뢰 책임자</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{reserv_name}</strong></td>
                        </tr>
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280; width:120px;"> 연락처(Phone number)</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{contact_phonenum}</strong></td>
                        </tr>
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280; width:120px;"> 이메일</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{contact_email}</strong></td>
                        </tr>
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280;">의뢰내용</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{reserv_purpose}</strong></td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>

            <!-- Help note -->
            <tr>
              <td style="padding:12px 24px 4px; font-family:Arial,Helvetica,sans-serif;">
                <p style="margin:0; font-size:13px; line-height:20px; color:#4b5563;">
                  변경이나 취소가 필요하시면 고객센터(<a href="mailto:help@spmed.kr" style="color:#16a34a; text-decoration:none;">help@spmed.kr</a>)로 연락 주시기 바랍니다.
                </p>
              </td>
            </tr>

            <!-- Divider -->
            <tr>
              <td style="padding:16px 24px 8px;">
                <hr style="border:none; border-top:1px solid #e5e7eb; margin:0;">
              </td>
            </tr>

            <!-- Footer -->
            <tr>
              <td align="center" style="padding:12px 24px 24px; font-family:Arial,Helvetica,sans-serif; color:#6b7280;">
                <p style="margin:0; font-size:12px; line-height:18px;">
                  © 2025 SPMED Co., Ltd.
                </p>
              </td>
            </tr>
          </table>
          <!-- /Container -->
        </td>
      </tr>
    </table>
    <!-- /Wrapper -->
  </body>
</html>

"""

  msg = MIMEText(body, "html", "utf-8")
  msg["Subject"] = subject
  msg["From"] = sender
  msg["To"] = recipient

  # 메일 전송
  try:
    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    server.starttls()  # TLS 보안 연결
    server.login(SMTP_USER, SMTP_PASS)
    server.sendmail(sender, [recipient], msg.as_string())
    server.quit()
    print("메일 전송 성공")
  except Exception as e:
    print("메일 전송 실패:", e)

  return json.dumps({"status": "ok", "message": "메일이 전송되었습니다."})

tools = [find_company_info, make_reservation, send_email]
# --------------------------------------------------------------
# ---------------------------- Prompt & LLM  ----------------------------

from pytz import timezone
kst = timezone('Asia/Seoul')
now_kst = datetime.now(kst)

SYSTEM_PROMPT = """
[중요 규칙]
1. 사용자가 이미 제공한 예약 정보는 절대 다시 묻지 말고, 누락된 항목만 질문할 것.
2. 도구 호출 전에는 불필요한 안내 문장을 출력하지 말 것.
3. 예약 정보 수집은 반드시 make_reservation tool을 통해서만 진행할 것.

[역할]
당신은 예약/회사정보 비서 챗봇이다.
입력 받은 언어 대로(영어 -> 영어, 한국어 -> 한국어) 대답하세요. 만약 사용자가 한국어 사용자일 때, 사용하는 데이터가 영어라도 한국어로 번역해서 대답해야합니다.

[도구 설명]
1. 회사 정보 검색 -> find_company_info
   - SPMED 관련 정보를 검색해 제공.
2. 서비스 신청(예약, 의뢰) -> make_reservation
   - 사용자의 예약/의뢰 정보를 수집.
3. 서비스 신청(예약, 의뢰) 확인 메일 발신 -> send_email
   - 사용자가 작성한 정보를 기반으로 메일 발신

[예약 절차]
1. 사용자가 회사 서비스 신청을 원하면, 연락처·팩스·주소를 find_company_info tool을 사용하여 해당 데이터를 가져온 후 안내한다. 이후 챗봇을 통해 의뢰 가능함을 알린다.
   ->ex) 회사 서비스 신청은 연락처 : ~~ / 팩스 : ~~ / 주소 : ~~ 를 통해서 직접 예약하실 수 있습니다. 또는 챗봇을 통해서도 예약할 수 있습니다.
2. 챗봇으로 서비스 신청을 원할 경우 make_reservation tool을 사용해 필요한 데이터를 수집한다. 데이터를 수집할 때, 임의로 작성하지 않으며, 모든 정보를 빠짐없이 물어봐야 한다. 처음 물어볼 경우는 더더욱 확실하게 물어봐야 한다.
3. 모든 필수 정보가 채워지면 사용자에게 입력된 정보가 맞는지 확인을 받는다. (중요)
4. 맞다고 확인하면 send_email tool로 이메일 발송과 동시에 데이터 베이스에 저장 (log_in_id State 내에 저장된 것을 쓰면 됩니다.).
5. 틀리거나 수정 요청 시 다시 make_reservation tool로 재수집 후 확인 절차 반복.

[예약 신청 사이클]
make_reservation -> post_tool_node -> agent -> (모든 정보가 잘 모였다면) send_email

마지막으로 답변들을 마크다운으로 각 항목을 번호를 붙여가며 쉽게 알아볼 수 있도록 작성하세요.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", f"현재 날짜 : {now_kst.strftime('%Y-%m-%d %A')}"),
    MessagesPlaceholder("messages"),
])
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)
chain_with_tools = prompt | llm_with_tools
memory = MemorySaver()

# --------------------------------------------------------------
# ---------------------------- node ----------------------------

from langchain_core.messages import AIMessage

def agent2(state: State2) -> Dict:
    ai = chain_with_tools.invoke({"messages": state["messages"]})
    new_state = {
        **state,  
        "messages": state["messages"] + [ai],  # 메시지만 누적
    }
    return new_state

def post_tool_node(state: State2) -> dict:
    import json, ast
    
    # 1. 가장 마지막 메시지(ToolMessage)와 그 이전 메시지(AIMessage)를 가져옵니다.
    last_message = state["messages"][-1]
    ai_message = state["messages"][-2]

    # 2. 마지막으로 호출된 도구의 이름을 확인합니다.
    tool_name = ""
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        tool_name = ai_message.tool_calls[0].get("name", "")

        if tool_name == "make_reservation" and isinstance(last_message, ToolMessage):
            try:
                tool_output = json.loads(last_message.content)
                # state 업데이트
                for k, v in tool_output.items():
                    if v:  # None/빈값 무시
                        state[k] = v
                print("✅ make_reservation result merged into state:", tool_output)
            except Exception as e:
                print(f"❌ Failed to merge make_reservation output: {e}")

    # 3. 만약 'send_email' 도구가 호출되었고, 성공했다면 DB에 저장합니다.
        if tool_name == "send_email" and isinstance(last_message, ToolMessage):
            try:
                # 도구 실행 결과가 성공적인지 확인 (JSON 파싱)
                tool_output = json.loads(last_message.content)
                if tool_output.get("status") == "ok":
                    print("✅ 'send_email' tool success detected. Saving reservation to DB...")
                    
                    # state에서 모든 예약 정보를 가져옵니다.
                    log_in_id = state.get("log_in_id")
                    reserv_company = state.get("reserv_company")
                    reserv_name = state.get("reserv_name")
                    contact_email = state.get("contact_email")
                    contact_phonenum = state.get("contact_phonenum")
                    reserv_purpose = state.get("reserv_purpose")

                    print([log_in_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose], " 1")

                    # 모든 정보가 있는지 한번 더 확인 후 저장
                    if all([log_in_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose]):
                        save_reserv(
                            log_in_id,
                            reserv_company,
                            reserv_name,
                            contact_email,
                            contact_phonenum,
                            reserv_purpose
                        )
                        print("✅ Reservation information saved to DB successfully.")
                    else:
                        print([log_in_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose], " 2")
                        print("❌ Not enough information in state to save the reservation.")

            except Exception as e:
                print(f"❌ Failed to save reservation to DB. Error: {e}")
    return state

# --------------------------------------------------------------
# ---------------------------- graph ---------------------------
builder2 = StateGraph(State2)

builder2.add_node("agent2", agent2)
builder2.add_node("tools", ToolNode(tools))

builder2.add_conditional_edges(
    "agent2",
    tools_condition,
    {"tools" : "tools", "end" : END, "__end__" : END}
)

builder2.set_entry_point("agent2")

builder2.add_node("post_tool", post_tool_node)
builder2.add_edge("tools", "post_tool")
builder2.add_edge("post_tool", "agent2")

graph2 = builder2.compile(checkpointer=memory)

#________________________ run code ________________________

def run_chat(user_message: str, session: dict) -> str:
    log_in_id = session["configurable"]["log_in_id"]
  
    # ✅ 1. 새로 추가한 함수를 여기서 호출합니다.
    reservation_data = load_latest_reservation(log_in_id)
    if reservation_data is None:
        reservation_data = {} # 정보가 없으면 빈 딕셔너리로 설정

    # 2. 대화 기록을 불러옵니다.
    prev_messages = load_chat_history(log_in_id)
    current_messages = prev_messages + [HumanMessage(content=user_message)]

    # 3. 사용자 메시지를 DB에 저장합니다.
    save_message(log_in_id, "human", user_message)

    input_state = {
    "messages": current_messages,
    "log_in_id": log_in_id,
    **reservation_data}

    # 5. LangGraph를 호출합니다.
    output_state = graph2.invoke(
        input_state,
        config=session
    )

    # 6. AI 답변을 처리합니다.
    final_msg = output_state["messages"][-1]
    if isinstance(final_msg, AIMessage):
        save_message(log_in_id, "ai", final_msg.content)
        return final_msg.content
        
    return ""