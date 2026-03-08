import asyncio
import base64
import time

import streamlit as st
from agents import (
    Agent,
    FileSearchTool,
    ImageGenerationTool,
    Runner,
    SQLiteSession,
    WebSearchTool,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# SQLiteSession이 image_generation_call 등의 결과를 저장할 때
# API output 형식 그대로 저장하는데, 이를 다음 턴 input으로 재전송하면
# 'action' 같은 input 스키마에 없는 필드가 포함돼 400 에러가 발생함.
# get_items() 오버라이드로 해당 필드를 제거해 재전송 전에 필터링함.
TOOL_CALL_TYPES = {"image_generation_call", "code_interpreter_call", "file_search_call", "web_search_call"}
INVALID_INPUT_FIELDS = {"action", "partial_images"}


class FilteredSQLiteSession(SQLiteSession):
    async def get_items(self):
        items = await super().get_items()
        filtered = []
        for item in items:
            if isinstance(item, dict) and item.get("type") in TOOL_CALL_TYPES:
                item = {k: v for k, v in item.items() if k not in INVALID_INPUT_FIELDS}
            filtered.append(item)
        return filtered


VECTOR_STORE_ID = "vs_69a8370429cc8191b0d03bb6665f9ddd"
APP_TITLE = "🌱 Life Coach"
APP_SUBTITLE = "당신의 목표, 진행 상황, 그리고 다음 한 걸음을 함께 설계하는 코치"

DEMO_PROMPTS = {
    "life_journey": """Analyze my current goals and create a Life Journey Visualization.
Use my uploaded files first if relevant.
Then:
1. summarize my main goal in one sentence,
2. identify my current stage,
3. extract 3 milestones,
4. define one next action,
5. call the ImageGenerationTool to create  a progress-roadmap image in a clean modern life-coaching app style.
Respond in Korean, but write the image generation prompt in English for the best quality.
DO NOT show he image generation promp on the screen.""",
    "journal_reflection": """Review my uploaded files and reflections, then perform a Journal Insight Analysis.
Use file search first.
Then:
1. identify 3 meaningful patterns in my progress,
2. explain what is helping me,
3. explain what is slowing me down,
4. suggest one practical next step,
5. end with a short encouraging coaching note.
Respond in Korean.""",
    "motivation_generator": """I feel stuck. Create an AI Motivation Boost for me.
Use my uploaded files first if relevant.
Then:
1. explain my current situation with empathy,
2. identify one meaningful reason not to quit,
3. suggest one small next action,
4. generate a motivational image or poster in a clean modern life-coaching style,
5. include only a very short motivational phrase inside the image.
Respond in Korean, but generate the image generation prompt in English.
DO NOT show he image generation promp on the screen.""",
}

AGENT_INSTRUCTIONS = """
You are a friendly, supportive, and insightful Life Coach AI.

Your goal is to help users reflect on their life, set meaningful goals, stay motivated, and make consistent progress toward their aspirations.

You help the user with:
- goal setting
- reflection
- habit building
- progress tracking
- personal growth

Always respond with empathy, encouragement, and thoughtful guidance.
Balance emotional support with practical next steps.
Respond in the same language the user writes in unless the user explicitly asks otherwise.

--------------------------------------------------
CORE CAPABILITIES
--------------------------------------------------

You are especially good at these three capabilities:

1. Life Journey Visualization
- analyze the user's goals
- identify current stage, milestones, final direction, and next action
- create a visual roadmap when helpful

2. Journal Insight Analysis
- analyze uploaded reflections, plans, journals, or notes
- identify patterns, momentum, blockers, and strengths
- summarize them as coaching insights

3. Motivation Generator
- when the user feels stuck, overwhelmed, or discouraged,
  provide empathetic coaching and a very small next step
- when helpful, create a motivational image or poster

--------------------------------------------------
MANDATORY TWO-STEP SEARCH PROCES
--------------------------------------------------

STEP 1 — File Search Tool (ALWAYS first)
Use the File Search Tool to check the user's uploaded documents.
This gives you personalized context: their goals, habits, progress, journals, and plans.
Never skip this step. Even for general questions, check files first.

STEP 2 — Web Search Tool (ALWAYS second)
After file search, ALWAYS use the Web Search Tool to supplement your answer.
Use this when the user asks a question that isn't in your training data.
Use this tool when the user asks about current or future events, when you think you don't know the answer, try searching for it on the web first.
Use web search to find:
- evidence-based methods and strategies
- tips, productivity systems, motivation techniques
- wellness advice, habit-building research
Never skip this step. Even if file search returned good results, always enrich with web search.

STEP 3 — Combine and respond
Merge both results into one cohesive, personalized, evidence-based answer.
- Use file search results to make advice personal and specific to the user.
- Use web search results to make advice credible and up-to-date.
- Never answer from your training data alone.
- Never use only one tool when both are applicable.

--------------------------------------------------
IMAGE GENERATION TOOL
--------------------------------------------------

You have access to an Image Generation Tool.

Use it when the user asks for visual inspiration or when a visual representation of their goals or progress would clearly improve the coaching experience.

Examples:
- vision board
- progress roadmap
- Life Journey Visualization
- motivational poster
- habit-growth visual
- celebration image
- future-self image

Before generating an image:
1. If the request relates to the user's goals, plans, habits, journals, reflections, or progress,
   first use the File Search Tool.
2. Extract concrete themes such as:
   - current stage
   - main goal
   - 2 to 4 milestones
   - next step
   - emotional tone
   - aspiration
3. Use those details in the image prompt.

--------------------------------------------------
IMAGE PROMPT CONSTRUCTION RULES
--------------------------------------------------

Always generate image prompts in English for best quality.

When constructing an image prompt, clearly describe:
1. subject
2. scene composition
3. symbolic elements
4. emotional tone
5. visual style
6. optional short text

Prefer symbolic visual metaphors such as:
- mountain climb
- staircase of progress
- road toward a goal flag
- roadmap with milestones
- stepping stones
- seed growing into a tree
- sunrise representing a new beginning
- glowing summit

Tone:
- uplifting
- hopeful
- motivating
- calm
- inspiring

--------------------------------------------------
IMAGE STYLE RULES
--------------------------------------------------

Preferred style:
- clean modern digital illustration
- minimal clutter
- elegant composition
- soft gradient background
- warm and encouraging colors
- app-friendly visual design
- modern life coaching app aesthetic
- modern SaaS-style illustration

Avoid:
- chaotic composition
- excessive text
- crowded scenes
- confusing symbolism
- dark or discouraging mood unless the user explicitly asks for it

--------------------------------------------------
TEXT INSIDE IMAGES
--------------------------------------------------

If text is included in the image, keep it very short.
Good examples:
- Keep Going
- Step by Step
- Goal in Progress
- You're Growing
- Small Steps
- Stay Consistent
- You Did It

Never include large blocks of text inside the image.

--------------------------------------------------
IMAGE TYPES
--------------------------------------------------

1. Vision Board
A visual representation of the user's future goals and aspirations.

2. Progress Roadmap
A journey showing:
- starting point
- current stage
- 2 to 4 milestones
- final goal
- next action

3. Celebration Poster
A motivational image for a meaningful achievement.

4. Habit Growth Visual
An image showing how small daily actions lead to long-term growth.

--------------------------------------------------
RESPONSE STYLE
--------------------------------------------------

Be supportive, clear, and insightful.
Celebrate effort, not only outcomes.
When useful, summarize your coaching in this order:
1. what you noticed
2. what it means
3. what to do next

When appropriate, suggest or generate a visual image that helps the user stay motivated.
Your role is to help the user design a meaningful life and keep moving forward.
"""


if "demo_prompt_queue" not in st.session_state:
    st.session_state["demo_prompt_queue"] = None

st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="centered")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions=AGENT_INSTRUCTIONS,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=5,
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "medium",
                    "output_format": "jpeg",
                    "moderation": "low",
                    "partial_images": 1,
                    "size": "1024x1024",
                }
            ),
        ],
    )

agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = FilteredSQLiteSession(
        "chat-history",
        "life-coach-memory.db",
    )

session = st.session_state["session"]


def queue_demo_prompt(prompt_key: str):
    st.session_state["demo_prompt_queue"] = DEMO_PROMPTS[prompt_key]


def update_status(status_container, event_type: str):
    status_messages = {
        "response.web_search_call.in_progress": ("🔍 웹 검색 시작 중...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색 중...", "running"),
        "response.web_search_call.completed": ("✅ 웹 검색 완료", "complete"),
        "response.file_search_call.in_progress": ("🗂️ 파일 검색 시작 중...", "running"),
        "response.file_search_call.searching": ("🗂️ 파일 검색 중...", "running"),
        "response.file_search_call.completed": ("✅ 파일 검색 완료", "complete"),
        "response.image_generation_call.in_progress": ("🎨 이미지 생성 준비 중...", "running"),
        "response.image_generation_call.generating": ("🎨 이미지 생성 중...", "running"),
        "response.completed": ("✅ 응답 완료", "complete"),
    }

    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            role = "assistant" if message["role"] == "assistant" else message["role"]
            avatar_role = "ai" if role == "assistant" else role
            with st.chat_message(avatar_role):
                if message["role"] == "user":
                    st.write(message["content"])
                elif message.get("type") == "message":
                    text = message["content"][0]["text"]
                    st.write(text.replace("$", r"\$"))

        if "type" in message:
            if message["type"] == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 웹을 검색했습니다.")
            elif message["type"] == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ 업로드한 파일을 확인했습니다.")
            elif message["type"] == "image_generation_call":
                with st.chat_message("ai"):
                    try:
                        image = base64.b64decode(message["result"])
                        st.image(image)
                    except Exception:
                        st.write("이미지를 불러오지 못했습니다.")


asyncio.run(paint_history())


async def run_agent(message: str):
    with st.chat_message("ai"):
        status_container = st.status("⏳ 생각 중...", expanded=False)
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response_text = ""

        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type != "raw_response_event":
                continue

            update_status(status_container, event.data.type)

            if event.data.type == "response.output_text.delta":
                response_text += event.data.delta
                text_placeholder.write(response_text.replace("$", r"\$"))
            elif event.data.type == "response.image_generation_call.partial_image":
                image = base64.b64decode(event.data.partial_image_b64)
                image_placeholder.image(image)


prompt = st.chat_input(
    "무엇이든 이야기해보세요. 목표, 진행상황, 저널 분석, 동기부여 이미지까지 도와드릴게요.",
    accept_file=True,
    file_type=["txt", "pdf"],
)

if prompt:

    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        with st.chat_message("ai"):
            with st.status("⏳ 파일 업로드 중...") as status:
                uploaded_file = client.files.create(
                    file=(file.name, file.getvalue()),
                    purpose="user_data",
                )
                status.update(label="⏳ 벡터 스토어에 저장 중...")
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
    st.subheader("Demo Features")
    st.write("아래 버튼으로 데모용 기능을 바로 실행할 수 있습니다.")

    st.button(
        "1) Life Journey Visualization",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("life_journey",),
    )
    st.button(
        "2) Journal Insight Analysis",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("journal_reflection",),
    )
    st.button(
        "3) Motivation Generator",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("motivation_generator",),
    )

    st.divider()
    st.subheader("Prompt Templates")
    st.code(DEMO_PROMPTS["life_journey"], language="text")
    st.code(DEMO_PROMPTS["journal_reflection"], language="text")
    st.code(DEMO_PROMPTS["motivation_generator"], language="text")

    st.divider()
    reset = st.button("Reset memory", use_container_width=True)
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["demo_prompt_queue"] = None
        st.rerun()
    st.write(asyncio.run(session.get_items()))


queued_prompt = st.session_state.get("demo_prompt_queue")
if queued_prompt:
    st.session_state["demo_prompt_queue"] = None
    with st.chat_message("human"):
        st.write(queued_prompt)
    asyncio.run(run_agent(queued_prompt))
