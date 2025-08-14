## Import Line ------------------------------------------------------

import os
import json
# from IPython.display import Image, display

# TypeAnotation
from typing_extensions import TypedDict, Literal, List, Optional, Any, Dict
from typing_extensions import Sequence, Annotated

import random

# Graph
from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Langchain_Core
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langchain rag
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber

from langchain.chains import RetrievalQA

# search tool
from langchain_community.tools.tavily_search import TavilySearchResults

## ----------------------------------------------------------------------

class State2(TypedDict):
  messages : Annotated[Sequence[BaseMessage], add_messages]
  reserv_company : str | None
  reserv_name : str | None
  contact_email : str | None
  contact_phonenum : str | None
  reserv_purpose : str | None
  reserv_memo : str | None
  # reserv_date : str | None

# ---------------------------- tool ----------------------------
@tool
def find_company_info(query: str) -> str:
    """SPMED 관련 문서를 검색해 상위 결과를 문자열로 반환합니다."""
    db = FAISS.load_local("/Users/kimgkangmin/Desktop/code/SPMAD/faiss_index",
    OpenAIEmbeddings(api_key=" *****"),
    allow_dangerous_deserialization=True
    )
    docs = db.as_retriever(k=3).invoke(query)
    if not docs:
        return ""
    return "Top matches:\n" + "\n".join(f"- {d.page_content}" for d in docs)


@tool
def make_reservation(
    reserv_company: Optional[str] = None,
    reserv_name: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_phonenum: Optional[str] = None,
    reserv_purpose: Optional[str] = None,
    reserv_memo: Optional[str] = None,
    state: Optional[dict] = None
) -> dict:
    """예약 정보를 받는 도구입니다. (부족하면 interrupt → resume 병합 후 정상처리)
    항상 state와 병합하여 모든 필드를 채운 상태로 동작합니다.
    """

    # ✅ 함수 인자 + state 병합 (state 우선)
    fields = {
        "reserv_company": state.get("reserv_company") if state and state.get("reserv_company") else reserv_company,
        "reserv_name": state.get("reserv_name") if state and state.get("reserv_name") else reserv_name,
        "contact_email": state.get("contact_email") if state and state.get("contact_email") else contact_email,
        "contact_phonenum": state.get("contact_phonenum") if state and state.get("contact_phonenum") else contact_phonenum,
        "reserv_purpose": state.get("reserv_purpose") if state and state.get("reserv_purpose") else reserv_purpose,
        "reserv_memo": state.get("reserv_memo") if state and state.get("reserv_memo") else reserv_memo,
    }

    required = ["reserv_company", "reserv_name", "contact_email", "contact_phonenum", "reserv_purpose"]

    # 누락 필드 전부 한 번에 interrupt
    missing = [k for k in required if not fields.get(k)]
    if missing:
        human = interrupt({
            "missing": missing,
            "schema": {k: "string" for k in missing},
            "note": "누락된 항목을 모두 입력해주세요.",
        })
        for k in missing:
            if human.get(k):
                fields[k] = human[k]

    # 최종 검증
    missing_final = [k for k in required if not fields.get(k)]
    if missing_final:
        return {"tool": "make_reservation", "status": "REJECTED", "missing": missing_final}

    # ✅ 여기서 state 업데이트 (LangGraph에선 post_tools 없이도 반영되게)
    if state is not None:
        for k, v in fields.items():
            state[k] = v

    # 정상 처리 결과
    return {
        "tool": "make_reservation",
        "status": "CONFIRMED",
        "company": fields["reserv_company"],
        "applicant_name": fields["reserv_name"],
        "contact_info": fields["contact_phonenum"] or fields["contact_email"],
        "purpose": fields["reserv_purpose"],
        "memo": fields["reserv_memo"],
    }

@tool
def send_email(state : State2) -> dict:
  """의뢰에 필요한 모든 필드가 체워지고, 의뢰 정보 확인을 받은 뒤, 의뢰 정보를 메일로 보내는 tool입니다."""
  reserv_company = state.get("reserv_company")
  reserv_name = state.get("reserv_name")
  contact_email = state.get("contact_email")
  contact_phonenum = state.get("contact_phonenum")
  reserv_purpose = state.get("reserv_purpose")
  reserv_memo = state.get("reserv_memo")

  import smtplib
  from email.mime.text import MIMEText

  # SMTP 서버 정보
  SMTP_HOST = "email-smtp.ap-northeast-2.amazonaws.com"
  SMTP_PORT = 587

  SMTP_USER = "AKIAV5W7O4PLVAVB5VMA" # AWS SES에서 발급
  SMTP_PASS = "BI3+c4nmjGyYQpfHRzyAJmtnFDietXWnqakQ//ouiP0u" # AWS SES에서 발급

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
                        <tr>
                          <td style="padding:6px 0; font-size:14px; color:#6b7280;">추가 정보</td>
                          <td style="padding:6px 0; font-size:14px; color:#111827;"><strong>{reserv_memo}</strong></td>
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

  return {"messages": [AIMessage(content="메일이 전송되었습니다.")]}



tools = [find_company_info, make_reservation, send_email]
# --------------------------------------------------------------
# ---------------------------- Prompt & LLM  ----------------------------

from datetime import datetime
from pytz import timezone
kst = timezone('Asia/Seoul')
now_kst = datetime.now(kst)

SYSTEM_PROMPT = """
[강제 규칙]
- 사용자가 예약/의뢰를 요청하면 반드시 make_reservation tool을 호출하고, 즉시 interruption을 발생시킬 것.
- 예약 정보를 직접 묻거나 채우지 말고, make_reservation tool의 인자로만 전달할 것.

[중요 규칙]
1. 사용자가 이미 제공한 예약 정보는 절대 다시 묻지 말고, 누락된 항목만 질문할 것.
2. 도구 호출 전에는 불필요한 안내 문장을 출력하지 말 것.
3. 예약 정보 수집은 반드시 make_reservation tool을 통해서만 진행할 것.

[역할]
당신은 예약/회사정보 비서 챗봇이다.

[도구 설명]
1. 회사 정보 검색 → find_company_info
   - SPMED 관련 정보를 검색해 제공.
2. 서비스 신청(예약, 의뢰) → make_reservation
   - 사용자의 예약/의뢰 정보를 수집.

[예약 절차]
1. 사용자가 회사 서비스 신청을 원하면, 연락처·팩스·주소를 find_company_info tool을 사용하여 해당 데이터를 가져온 후 안내한다. 이후 챗봇을 통해 의뢰 가능함을 알린다.
2. 챗봇으로 서비스 신청을 원할 경우 make_reservation tool을 사용해 필요한 데이터를 수집한다.
3. interruption 발생 시, 누락된 항목만 물어본다.
4. 모든 필수 정보가 채워지면 사용자에게 입력된 정보가 맞는지 확인을 받는다. (중요)
5. 맞다고 확인하면 send_email tool로 이메일 발송.
6. 틀리거나 수정 요청 시 다시 make_reservation tool로 재수집 후 확인 절차 반복.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", f"현재 날짜 : {now_kst.strftime('%Y-%m-%d %A')}"),
    MessagesPlaceholder("messages"),
])
llm = ChatOpenAI(model="gpt-4o", api_key=" ***** ")
llm_with_tools = llm.bind_tools(tools)
chain_with_tools = prompt | llm_with_tools

memory = MemorySaver()

# --------------------------------------------------------------
# ---------------------------- node ----------------------------

from langchain_core.messages import AIMessage

def agent2(state):
    msgs = state["messages"]

    # 🔒 이미 tool_calls가 대기 중이면, LLM을 다시 호출하면 안 됨
    if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
        return {}  # 그대로 ToolNode로 넘어가게 둠

    ai = chain_with_tools.invoke({"messages": msgs})
    return {"messages": [ai]}  # 새 AI 메시지 '한 개'만 추가


def pre_tool_node(state: State2):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    for tc in tool_calls:
        if tc.get("name") == "make_reservation":
            args = dict(tc.get("args") or {})

            # state 값 채우기
            for k in [
                "reserv_company",
                "reserv_name",
                "contact_email",
                "contact_phonenum",
                "reserv_purpose",
                "reserv_memo"
            ]:
                if not args.get(k) and state.get(k):
                    args[k] = state[k]

            state = {**state, **{k: v for k, v in args.items() if v}}

            tc["args"] = args

    return state


def post_tools(state):
    last = state["messages"][-1]
    if not isinstance(last, ToolMessage) or getattr(last, "name", "") != "make_reservation":
        return {}

    data = last.content
    if isinstance(data, str):
        try: data = json.loads(data)
        except: return {}

    updates = {}
    if v := data.get("company"):         updates["reserv_company"] = v
    if v := data.get("applicant_name"):  updates["reserv_name"] = v
    if v := data.get("purpose"):         updates["reserv_purpose"] = v
    if v := data.get("memo"):            updates["reserv_memo"] = v
    if (ci := data.get("contact_info")):
        updates["contact_email" if any(c.isalpha() for c in ci) else "contact_phonenum"] = ci

    return {**state, **updates}


# --------------------------------------------------------------
# ---------------------------- graph ---------------------------
builder2 = StateGraph(State2)

builder2.add_node("agent2", agent2)
builder2.add_node("pre_tools", pre_tool_node)   # ← 추가
builder2.add_node("tools", ToolNode(tools))
builder2.add_node("post_tools", post_tools)

builder2.add_conditional_edges(
    "agent2",
    tools_condition,
    {"tools": "pre_tools", "end": END, "__end__": END}  # ← tools 대신 pre_tools로
)

builder2.set_entry_point("agent2")

builder2.add_edge("pre_tools", "tools")
builder2.add_edge("tools", "post_tools")
builder2.add_edge("post_tools", "agent2")

graph2 = builder2.compile(checkpointer=memory)
# --------------------------------------------------------------

 # === 필요한 import ===
# from typing import Optional, Dict, Any
# from langgraph.types import interrupt, Command
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- 인터럽트가 뜨면 콘솔에서 필요한 값만 받아 resume ---
# === 인터럽트 핸들러 ===

def on_interrupt_console(_snap) -> Dict[str, Any]:
    """
    인터럽트 발생 시, 현재 state에서 빈 필드만 질문하여
    '툴 인자 이름'과 동일한 키로 값을 반환합니다.
    """
    # _snap.values가 있으면 거기서, 없으면 dict 변환
    values = getattr(_snap, "values", None)
    if values is None:
        values = dict(_snap) if not isinstance(_snap, dict) else _snap

    allowed = [
        "reserv_company", "reserv_name", "contact_email",
        "contact_phonenum", "reserv_purpose", "reserv_memo"
    ]
    labels = {
        "reserv_company":   "의뢰사/기관",
        "reserv_name":      "의뢰 책임자 성함",
        "contact_email":    "이메일",
        "contact_phonenum": "연락처(전화번호)",
        "reserv_purpose":   "의뢰 내용",
        "reserv_memo":      "추가 정보(선택)",
    }

    # 🔒 스냅에 missing 정보가 없어도 안전하게: state에서 직접 빈 필드 탐지
    missing = [k for k in allowed if not values.get(k)]

    print("\n[HITL] 의뢰서 작성을 위해 추가 정보가 필요합니다. (Enter로 건너뛰기)")
    out: Dict[str, Any] = {}
    for field in missing:
        prev = values.get(field)
        prompt = f" - {labels.get(field, field)}({field})"
        if prev:
            prompt += f" [{prev}]"
        prompt += ": "

        val = input(prompt).strip()
        if val:
            out[field] = val
    return out


# === 한 턴 실행 ===
def run_one_turn(graph, user_text: str, cfg, on_interrupt):
    last_state = None
    last_len = 0
    cmd = {"messages": [HumanMessage(content=user_text)]}
    interrupt_count = 0

    while True:
        for state in graph.stream(cmd, cfg, stream_mode="values"):
            last_state = state
            msgs = state.get("messages", [])
            for m in msgs[last_len:]:
                if isinstance(m, AIMessage):
                    print("\n[Assistant]", m.content)
                elif isinstance(m, ToolMessage):
                    snippet = str(m.content)
                    print("\n[Tool]", snippet[:200] + ("..." if len(snippet) > 200 else ""))
            last_len = len(msgs)

        snap = graph.get_state(cfg)
        if not getattr(snap, "next", None):
            break  # 그래프 종료

        print("interrupt 발생")
        interrupt_count += 1
        if interrupt_count > 5:
            print("⚠️ 인터럽트가 계속 발생합니다. 툴의 병합 로직이나 키 이름을 확인하세요.")
            break

        resume_payload = on_interrupt(snap)
        if not resume_payload:
            print("⚠️ 입력이 비었습니다. 같은 누락이 재발할 수 있습니다.")
        cmd = Command(resume=resume_payload)

    if last_state and last_state.get("messages"):
        final = last_state["messages"][-1]
        if isinstance(final, AIMessage):
            print("\n[Final]", final.content)

# === 대화 루프 ===
def chat_loop(graph, cfg):
    print("대화를 시작합니다. 종료하려면 0 또는 /exit 를 입력하세요.")

    # 초기 State (State2에 맞게 None 세팅)
    init_state = {
        "messages": [],
        "reserv_company": None,
        "reserv_name": None,
        "contact_email": None,
        "contact_phonenum": None,
        "reserv_purpose": None,
        "reserv_memo": None,
    }

    # 초기화 실행
    _ = graph.stream(init_state, cfg, stream_mode="values")

    while True:
        user = input("[You] ").strip()
        if user in ("0", "/exit", "/quit"):
            print("종료합니다.")
            break
        run_one_turn(graph, user, cfg, on_interrupt_console)

# === 실행 ===
session = {"configurable": {"thread_id": "test1"}}
chat_loop(graph2, session)
