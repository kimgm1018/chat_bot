## Import Line ------------------------------------------------------

import os
import json

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
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langchain rag
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# DB
import sqlite3
from datetime import datetime

## langsmith -----------------------------------------------------------
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = " *** "
# os.environ["LANGCHAIN_PROJECT"] = "default"
# os.environ["OPENAI_API_KEY"] = " *** "
## ----------------------------------------------------------------------

class State2(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  user_id: str|None
  reserv_company: Annotated[Optional[str], lambda x, y: y or x]
  reserv_name: Annotated[Optional[str], lambda x, y: y or x]
  contact_email: Annotated[Optional[str], lambda x, y: y or x]
  contact_phonenum: Annotated[Optional[str], lambda x, y: y or x]
  reserv_purpose: Annotated[Optional[str], lambda x, y: y or x]

#____________________________________ DB ____________________________________

def create_user_table_if_not_exists(user_id: str):
    # user_id에 특수문자가 없는지 간단히 확인 (SQL 인젝션 방지)
    if not user_id.isalnum():
        raise ValueError("Invalid user_id format. Only alphanumeric characters are allowed.")
    
    with sqlite3.connect("chat_memory_1.db") as conn:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chat_{user_id} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        # 예약 테이블은 모든 유저가 공용으로 사용하므로 한번만 생성되도록 유지
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reserv_memory_1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            reserv_company TEXT NOT NULL,
            reserv_name TEXT NOT NULL,
            contact_email TEXT NOT NULL,
            contact_phonenum TEXT NOT NULL,
            reserv_purpose TEXT NOT NULL
        )
        """)
        conn.commit()

def save_message(user_id: str, role: str, content: str):
    if not user_id.isalnum():
        raise ValueError("Invalid user_id")
    create_user_table_if_not_exists(user_id) # 테이블이 없을 경우를 대비해 호출
    with sqlite3.connect("chat_memory_1.db") as conn:
        cur = conn.cursor()
        # ❗ f-string으로 동적 테이블 이름을 사용
        cur.execute(
            f"""INSERT INTO chat_{user_id} (user_id, role, content) VALUES (?, ?, ?)""",
            (user_id, role, content)
        )
        conn.commit()

def save_reserv(user_id: str, reserv_company: str, reserv_name: str, contact_email: str, contact_phonenum: str, reserv_purpose: str):
    create_user_table_if_not_exists(user_id) # 테이블 생성 확인
    with sqlite3.connect("chat_memory_1.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reserv_memory_1 
            (user_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose))
        conn.commit()

def load_chat_history(user_id: str):
    if not user_id.isalnum():
        raise ValueError("Invalid user_id")
    create_user_table_if_not_exists(user_id) # 테이블이 없을 경우를 대비해 호출
    with sqlite3.connect("chat_memory_1.db") as conn:
        cur = conn.cursor()
        try:
            # ❗ f-string으로 동적 테이블 이름을 사용
            cur.execute(f"""
                SELECT role, content 
                FROM chat_{user_id} 
                WHERE user_id = ? 
                ORDER BY timestamp ASC
            """, (user_id,))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # 테이블이 없는 경우 등 예외 발생 시 빈 리스트 반환
            return []
    
    messages = []
    for role, content in rows:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(ToolMessage(content=content, tool_call_id="dummy"))
    return messages

def load_latest_reservation(user_id: str) -> Optional[Dict[str, str]]:
    """가장 최근의 예약 정보를 DB에서 불러옵니다."""
    with sqlite3.connect("chat_memory_1.db") as conn:
        conn.row_factory = sqlite3.Row # 딕셔너리 형태로 결과를 받기 위함
        cur = conn.cursor()
        cur.execute("""
            SELECT reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose
            FROM reserv_memory_1
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = cur.fetchone()

    if row:
        return dict(row) # 딕셔너리로 변환하여 반환
    return None
#______________________________ DB ____________________________________


# ---------------------------- Tool ----------------------------
@tool
def find_company_info(query: str) -> str:
    """SPMED(회사) 관련 문서를 검색해 상위 결과를 문자열로 반환합니다."""
    db = FAISS.load_local("/Users/kimgkangmin/Desktop/code/SPMAD/faiss_index",
    OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),
    allow_dangerous_deserialization=True)
    docs = db.as_retriever(k=3).invoke(query)
    if not docs:
        return ""
    return "Top matches:\n" + "\n".join(f"- {d.page_content}" for d in docs)

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
  import sqlite3
  from langchain_core.messages import AIMessage

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

from datetime import datetime
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
4. 맞다고 확인하면 send_email tool로 이메일 발송과 동시에 데이터 베이스에 저장 (user_id는 State 내에 저장된 것을 쓰면 됩니다.).
5. 틀리거나 수정 요청 시 다시 make_reservation tool로 재수집 후 확인 절차 반복.

마지막으로 답변들을 마크다운으로 보기 쉽고 깔끔하게 작성하세요.
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
        **state,  # 기존 user_id, reserv_xx 다 유지
        "messages": state["messages"] + [ai],  # 메시지만 누적
    }
      # 여기서 user_id 찍어보면 바로 "test-1" 나와야 함
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

    # 3. 만약 'send_email' 도구가 호출되었고, 성공했다면 DB에 저장합니다.
    if tool_name == "send_email" and isinstance(last_message, ToolMessage):
        try:
            # 도구 실행 결과가 성공적인지 확인 (JSON 파싱)
            tool_output = json.loads(last_message.content)
            if tool_output.get("status") == "ok":
                print("✅ 'send_email' tool success detected. Saving reservation to DB...")
                
                # state에서 모든 예약 정보를 가져옵니다.
                user_id = state.get("user_id")
                reserv_company = state.get("reserv_company")
                reserv_name = state.get("reserv_name")
                contact_email = state.get("contact_email")
                contact_phonenum = state.get("contact_phonenum")
                reserv_purpose = state.get("reserv_purpose")

                print([user_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose], " 1")

                # 모든 정보가 있는지 한번 더 확인 후 저장
                if all([user_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose]):
                    save_reserv(
                        user_id,
                        reserv_company,
                        reserv_name,
                        contact_email,
                        contact_phonenum,
                        reserv_purpose
                    )
                    print("✅ Reservation information saved to DB successfully.")
                else:
                    print([user_id, reserv_company, reserv_name, contact_email, contact_phonenum, reserv_purpose], " 2")
                    print("❌ Not enough information in state to save the reservation.")

        except Exception as e:
            print(f"❌ Failed to save reservation to DB. Error: {e}")
    return {}

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
    user_id = session["configurable"]["user_id"]

    conn = sqlite3.connect("chat_memory_1.db")
    prev_messages = load_chat_history(user_id)
    if not prev_messages:
        prev_messages = []

    current_messages = prev_messages + [HumanMessage(content=user_message)]

    print(session)

    save_message(user_id, "human", user_message)
  
    input_state = {
      "messages": current_messages,
      "user_id": user_id
    }

  # LangGraph 호출
    output_state = graph2.invoke(
        input_state,
        config=session
    )

    # 마지막 메시지가 AI 답변이면 DB에 저장 후 반환
    final_msg = output_state["messages"][-1]
    if isinstance(final_msg, AIMessage):
        save_message(user_id, "ai", final_msg.content)
        return final_msg.content
    return ""

# def run_chat(user_message: str, session: dict) -> str:
#     user_id = session["configurable"]["user_id"]

#     conn = sqlite3.connect("chat_memory_1.db")
#     prev_messages = load_chat_history(user_id)
#     if not prev_messages:
#         prev_messages = []

#     current_messages = prev_messages + [HumanMessage(content=user_message)]

#     print(session)

#     save_message(user_id, "human", user_message)
  
#     input_state = {
#       "messages": current_messages,
#       "user_id": user_id
#     }

#   # LangGraph 호출
#     output_state = graph2.invoke(
#         input_state,
#         config=session
#     )

#     # 마지막 메시지가 AI 답변이면 DB에 저장 후 반환
#     final_msg = output_state["messages"][-1]
#     if isinstance(final_msg, AIMessage):
#         save_message(user_id, "ai", final_msg.content)
#         return final_msg.content
#     return ""

def run_chat(user_message: str, session: dict) -> str:
    user_id = session["configurable"]["user_id"]

    # ✅ 1. 새로 추가한 함수를 여기서 호출합니다.
    reservation_data = load_latest_reservation(user_id)
    if reservation_data is None:
        reservation_data = {} # 정보가 없으면 빈 딕셔너리로 설정

    # 2. 대화 기록을 불러옵니다.
    prev_messages = load_chat_history(user_id)
    current_messages = prev_messages + [HumanMessage(content=user_message)]

    # 3. 사용자 메시지를 DB에 저장합니다.
    save_message(user_id, "human", user_message)
  
    # ✅ 4. state에 불러온 예약 정보를 포함하여 LangGraph를 시작합니다.
    input_state = {
      "messages": current_messages,
      "user_id": user_id,
      **reservation_data # 딕셔너리를 풀어 state에 병합 (예: "reserv_company": "값")
    }

    # 5. LangGraph를 호출합니다.
    output_state = graph2.invoke(
        input_state,
        config=session
    )

    # 6. AI 답변을 처리합니다.
    final_msg = output_state["messages"][-1]
    if isinstance(final_msg, AIMessage):
        save_message(user_id, "ai", final_msg.content)
        return final_msg.content
        
    return ""

# --- 실행 루프 ---

# session = {"configurable": {"thread_id": USER_ID, "user_id": USER_ID}}

# _________________________graph init____________________________

# init_state: State2 = {
#     "messages": [],
#     "user_id": USER_ID,
#     "reserv_company": None,
#     "reserv_name": None,
#     "contact_email": None,
#     "contact_phonenum": None,
#     "reserv_purpose": None,
# }
# graph2.stream(init_state, session)
# graph2.update_state(
#     config=session,
#     values={"user_id": USER_ID}   # 여기에 원하는 key 전부
# )

# # _______________________________________________________________


# while True:
#     msg = input("[You] : ")
#     if msg in ("0", "/exit"):
#         break

#     reply = run_chat(msg, session)
#     print("[AI]", reply)
