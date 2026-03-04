import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI()

VECTOR_STORE_ID = "vs_69a8370429cc8191b0d03bb6665f9ddd"

st.title("🌱 Life Coach")

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions="""
        You are an encouraging and supportive life coach. Your role is to help users achieve their goals,
        build positive habits, and overcome challenges in their daily life.

        You have access to the following tools:
            - Web Search Tool: Use this when the user asks a question that isn't in your training data. Use this tool when the user asks about current or future events, when you think you don't know the answer, try searching for it on the web first.
            - File Search Tool: Use this tool when the user asks about their personal goals, diary entries, or progress. Search their uploaded documents to provide personalized coaching advice.

        CRITICAL RULES:
        1. When the user asks about their goals, progress, or personal records, ALWAYS use the File Search Tool first to reference their uploaded documents.
        2. You MUST use the Web Search Tool before answering ANY of the following:
           - Tips, methods, or strategies on any topic
           - How to build or break habits
           - Productivity, motivation, or wellness advice
           - Any practical, actionable recommendation
        3. Combine file search results with web search results to provide personalized, evidence-based recommendations.
        NEVER answer practical questions from your training data alone. ALWAYS search the web first.

        Guidelines:
        - Always be warm, empathetic, and motivating in your responses.
        - Celebrate the user's efforts and progress, no matter how small.
        - Provide practical, actionable advice based on your web search results.
        - Reference the user's uploaded goals and diary entries when giving personalized advice.
        - Ask follow-up questions to better understand the user's goals and challenges.
        - Respond in the same language the user writes in.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
        ],
    )

agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "life-coach-memory.db",
    )

session = st.session_state["session"]


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ 웹 검색 완료.", "complete"),
        "response.web_search_call.in_progress": ("🔍 웹 검색 시작 중...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색 중...", "running"),
        "response.file_search_call.completed": ("✅ 파일 검색 완료.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ 파일 검색 시작 중...", "running"),
        "response.file_search_call.searching": ("🗂️ 파일 검색 중...", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", r"\$"))
        if "type" in message:
            if message["type"] == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 웹을 검색했습니다...")
            elif message["type"] == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ 목표 문서를 검색했습니다...")


asyncio.run(paint_history())


async def run_agent(message):
    status_container = st.status("⏳ 생각 중...", expanded=False)
    text_placeholder = st.empty()
    response = ""

    stream = Runner.run_streamed(
        agent,
        message,
        session=session,
    )

    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            update_status(status_container, event.data.type)

            if event.data.type == "response.output_text.delta":
                response += event.data.delta
                text_placeholder.write(response.replace("$", r"\$"))


prompt = st.chat_input(
    "무엇이든 이야기해보세요. 코치가 함께합니다!",
    accept_file=True,
    file_type=["txt", "pdf"],
)

if prompt:
    for file in prompt.files:
        with st.chat_message("ai"):
            with st.status("⏳ 파일 업로드 중...") as status:
                uploaded_file = client.files.create(
                    file=(file.name, file.getvalue()),
                    purpose="user_data",
                )
                status.update(label="⏳ 목표 문서 저장 중...")
                client.vector_stores.files.create(
                    vector_store_id=VECTOR_STORE_ID,
                    file_id=uploaded_file.id,
                )
                time.sleep(3)
                status.update(label=f"✅ '{file.name}' 업로드 완료", state="complete")

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))

with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.rerun()
    st.write(asyncio.run(session.get_items()))
