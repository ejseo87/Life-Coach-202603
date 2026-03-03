import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession, WebSearchTool
from dotenv import load_dotenv

load_dotenv()

st.title("🌱 Life Coach")

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions="""
        You are an encouraging and supportive life coach. Your role is to help users achieve their goals,
        build positive habits, and overcome challenges in their daily life.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't in your training data. Use this tool when the users asks about current or future events, when you think you don't know the answer, try searching for it in the web first.

        CRITICAL RULE: You MUST use the Web Search Tool before answering ANY of the following:
        - Tips, methods, or strategies on any topic
        - How to build or break habits
        - Productivity, motivation, or wellness advice
        - Any practical, actionable recommendation
        NEVER answer practical questions from your training data alone. ALWAYS search the web first.

        Guidelines:
        - Always be warm, empathetic, and motivating in your responses.
        - Celebrate the user's efforts and progress, no matter how small.
        - Provide practical, actionable advice based on your web search results.
        - Ask follow-up questions to better understand the user's goals and challenges.
        - Respond in the same language the user writes in.
        """,
        tools=[
            WebSearchTool(),
        ],
    )

agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "life-coach-memory.db",
    )

session = st.session_state["session"]


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
                    query = (message.get("action") or {}).get("query")
                    if query:
                        st.write(f'🔍 웹 검색: "{query}"')
                    else:
                        st.write("🔍 웹을 검색했습니다...")


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
            event_type = event.data.type

            if event_type == "response.output_item.added":
                item = getattr(event.data, "item", None)
                if getattr(item, "type", None) == "web_search_call":
                    status_container.update(
                        label="🔍 웹 검색 중...", state="running"
                    )

            elif event_type == "response.output_item.done":
                item = getattr(event.data, "item", None)
                if getattr(item, "type", None) == "web_search_call":
                    action = getattr(item, "action", None)
                    query = getattr(action, "query", None)
                    if not query:
                        queries = getattr(action, "queries", None)
                        query = queries[0] if queries else None
                    label = f'✅ 웹 검색 완료: "{query}"' if query else "✅ 웹 검색 완료."
                    status_container.update(label=label, state="complete")

            elif event_type == "response.completed":
                status_container.update(label=" ", state="complete")

            elif event_type == "response.output_text.delta":
                response += getattr(event.data, "delta", "")
                text_placeholder.write(response.replace("$", r"\$"))


prompt = st.chat_input("무엇이든 이야기해보세요. 코치가 함께합니다!")

if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))

with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.rerun()
    st.write(asyncio.run(session.get_items()))
