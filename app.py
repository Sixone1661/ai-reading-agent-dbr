"""Streamlit V1.0 app for an AI Socratic reading support prototype."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from database import (
    add_message,
    create_session,
    get_messages,
    get_reading_note,
    get_sessions,
    init_db,
    rows_to_dicts,
    upsert_reading_note,
)
from export_utils import dataframe_to_csv_bytes, get_messages_dataframe, get_reading_notes_dataframe
from extract_utils import UnsupportedFileTypeError, extract_text_from_upload

APP_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = APP_DIR / "prompts" / "system_prompt.md"
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_CONTEXT_CHARS = 12000


def get_config_value(name: str, default: str = "") -> str:
    """Read config from local .env first, then Streamlit Cloud secrets."""
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default
    return str(secret_value).strip() if secret_value else default

load_dotenv(APP_DIR / ".env")
init_db()

st.set_page_config(
    page_title="AI Reading Agent DBR",
    page_icon="📚",
    layout="wide",
)

PRIMARY_COLOR = "#2563EB"
PAGE_BACKGROUND = "#F8FAFC"
CARD_BACKGROUND = "#FFFFFF"
BORDER_COLOR = "#E5E7EB"
TEXT_COLOR = "#1F2937"
MUTED_TEXT_COLOR = "#6B7280"


def inject_global_css() -> None:
    """Apply lightweight product styling without changing Streamlit behavior."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {PAGE_BACKGROUND};
            color: {TEXT_COLOR};
        }}
        h1, h2, h3 {{
            color: {TEXT_COLOR};
            letter-spacing: 0;
        }}
        [data-testid="stSidebar"] {{
            background: #EEF3F8;
            border-right: 1px solid {BORDER_COLOR};
        }}
        [data-testid="stMetric"] {{
            background: {CARD_BACKGROUND};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }}
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            border-radius: 8px;
            border: 1px solid {PRIMARY_COLOR};
            background: {PRIMARY_COLOR};
            color: white;
            font-weight: 600;
        }}
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: #1D4ED8;
            background: #1D4ED8;
            color: white;
        }}
        .role-hero {{
            max-width: 980px;
            margin: 2.2rem auto 1.2rem auto;
            text-align: center;
        }}
        .role-title {{
            font-size: 2.2rem;
            line-height: 1.2;
            font-weight: 760;
            color: {TEXT_COLOR};
            margin-bottom: 0.5rem;
        }}
        .role-subtitle {{
            font-size: 1.08rem;
            color: {MUTED_TEXT_COLOR};
            margin: 0 auto;
            max-width: 760px;
        }}
        .role-card-copy {{
            min-height: 126px;
        }}
        .role-card-title {{
            font-size: 1.35rem;
            font-weight: 720;
            color: {TEXT_COLOR};
            margin-bottom: 0.55rem;
        }}
        .role-card-desc {{
            color: {MUTED_TEXT_COLOR};
            line-height: 1.7;
            font-size: 0.98rem;
        }}
        .section-label {{
            color: {PRIMARY_COLOR};
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }}
        .muted-copy {{
            color: {MUTED_TEXT_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def switch_role(role: str | None) -> None:
    """Update the selected role and rerun the app."""
    st.session_state["role"] = role
    st.rerun()


def render_role_selection() -> None:
    """Render the product-style role selection landing page."""
    st.markdown(
        """
        <div class="role-hero">
            <div class="role-title">AI 文献阅读支架</div>
            <div class="role-subtitle">
                面向研究生批判性文献阅读能力提升的苏格拉底式对话工具
            </div>
            <p class="role-subtitle" style="margin-top: 1rem; font-size: 0.98rem;">
                请选择你的使用身份，系统将根据不同角色提供对应的阅读支持与管理功能。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, student_col, teacher_col, _ = st.columns([0.8, 1.5, 1.5, 0.8], gap="large")
    with student_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="role-card-copy">
                    <div class="role-card-title">我是学生</div>
                    <div class="role-card-desc">
                        上传课程文献，接受 AI 苏格拉底式提问，完成批判性阅读笔记。
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("进入学生端", use_container_width=True, key="enter_student"):
                switch_role("student")

    with teacher_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="role-card-copy">
                    <div class="role-card-title">我是教师</div>
                    <div class="role-card-desc">
                        查看学生阅读过程、对话记录与批判性笔记，支持 DBR 研究过程管理。
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("进入教师端", use_container_width=True, key="enter_teacher"):
                switch_role("teacher")


def load_system_prompt() -> str:
    """Load the Socratic tutor instruction used for every chat."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def get_openai_client() -> OpenAI | None:
    """Create an OpenAI-compatible client when an API key is configured."""
    api_key = get_config_value("OPENAI_API_KEY")
    base_url = get_config_value("OPENAI_BASE_URL")

    if not api_key:
        return None

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def build_context_prompt(session: dict) -> str:
    """Add local reading-task context without asking the model to summarize for the student."""
    excerpt = session.get("literature_excerpt") or "学生尚未提供摘要或正文节选。"
    if len(excerpt) > MAX_CONTEXT_CHARS:
        excerpt = (
            excerpt[:MAX_CONTEXT_CHARS]
            + "\n\n[系统提示：原文较长，此处仅截取前 12000 字符作为当前对话上下文。]"
        )
    return f"""
当前阅读任务信息：
- 学生编号：{session['student_id']}
- 周次节点：{session['week_node']}
- 文献题目：{session['literature_title']}
- 任务类型：{session['task_type']}
- 文献摘要或正文节选：{excerpt}

请基于以上信息开展苏格拉底式追问。不要替学生直接总结文献。
""".strip()


def call_ai(session: dict, history: list[dict]) -> str:
    """Call OpenAI Chat Completions and return the assistant response."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("未检测到 OPENAI_API_KEY。请先根据 README 配置 .env 文件。")

    model = get_config_value("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "system", "content": build_context_prompt(session)},
    ]
    for item in history:
        role = "assistant" if item["role"] == "assistant" else "user"
        messages.append({"role": role, "content": item["content"]})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content or "我想先请你再补充一点：你目前最困惑的是研究问题、方法，还是证据链？"


def ensure_session_state() -> None:
    """Initialize Streamlit session state keys."""
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("active_session", None)
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("literature_input_mode", "手动复制粘贴")
    st.session_state.setdefault("uploaded_literature_text", "")
    st.session_state.setdefault("uploaded_literature_name", "")


def render_student_page() -> None:
    """Student workflow: create reading session, chat, and submit reading notes."""
    st.header("学生端：AI 苏格拉底式文献阅读支架")
    st.caption("V1.0 原型：用于研究生课程文献阅读节点，保存对话与批判性阅读笔记。")

    with st.sidebar:
        if st.button("返回角色选择", use_container_width=True):
            switch_role(None)
        st.divider()
        st.subheader("阅读会话设置")
        input_mode = st.radio(
            "文献内容输入方式",
            ["手动复制粘贴", "上传 PDF / Word"],
            key="literature_input_mode",
            horizontal=True,
        )

        uploaded_literature_text = ""
        uploaded_literature_name = ""
        if input_mode == "上传 PDF / Word":
            uploaded_file = st.file_uploader(
                "上传 PDF 或 Word 文献",
                type=["pdf", "docx"],
                help="支持可复制文本型 PDF 和 .docx 文件；扫描版 PDF 可能无法抽取正文。",
            )
            if uploaded_file is None:
                st.session_state.uploaded_literature_text = ""
                st.session_state.uploaded_literature_name = ""
                st.info("请选择一个 PDF 或 .docx 文件。")
            else:
                try:
                    uploaded_literature_text = extract_text_from_upload(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                    st.session_state.uploaded_literature_text = uploaded_literature_text
                    st.session_state.uploaded_literature_name = uploaded_file.name
                    if uploaded_literature_text:
                        st.success(f"已从 {uploaded_file.name} 抽取 {len(uploaded_literature_text)} 个字符。")
                        st.text_area(
                            "抽取文本预览，可创建会话后进入对话",
                            value=uploaded_literature_text[:3000],
                            height=160,
                            disabled=True,
                        )
                    else:
                        st.warning("没有抽取到文本。若是扫描版 PDF，请先 OCR，或改用手动粘贴。")
                except UnsupportedFileTypeError as exc:
                    st.session_state.uploaded_literature_text = ""
                    st.session_state.uploaded_literature_name = ""
                    st.error(f"文件解析失败：{exc}")
                except Exception as exc:  # Keep uploads friendly in classroom pilots.
                    st.session_state.uploaded_literature_text = ""
                    st.session_state.uploaded_literature_name = ""
                    st.error(f"文件解析失败：{exc}")

        with st.form("session_form"):
            student_id = st.text_input("学生编号", placeholder="如 S001")
            week_node = st.selectbox("周次节点", ["T0", "T1", "T2", "T3"])
            literature_title = st.text_input("文献题目")
            if input_mode == "手动复制粘贴":
                literature_excerpt = st.text_area("文献摘要或正文节选", height=160)
            else:
                literature_excerpt = st.session_state.uploaded_literature_text
                if st.session_state.uploaded_literature_name:
                    st.caption(f"当前上传文件：{st.session_state.uploaded_literature_name}")
            task_type = st.selectbox(
                "任务类型",
                [
                    "研究问题分析",
                    "理论框架分析",
                    "研究方法分析",
                    "证据链分析",
                    "局限与贡献分析",
                    "迁移反思",
                    "综合批判性阅读",
                ],
            )
            submitted = st.form_submit_button("开始新的阅读会话")
        if submitted:
            if not student_id or not literature_title:
                st.error("请至少填写学生编号和文献题目。")
            elif input_mode == "上传 PDF / Word" and not literature_excerpt.strip():
                st.error("请先上传可解析的 PDF 或 Word 文献，或切换为手动复制粘贴。")
            else:
                session_id = create_session(
                    student_id.strip(),
                    week_node,
                    literature_title.strip(),
                    literature_excerpt.strip(),
                    task_type,
                )
                session = {
                    "id": session_id,
                    "student_id": student_id.strip(),
                    "week_node": week_node,
                    "literature_title": literature_title.strip(),
                    "literature_excerpt": literature_excerpt.strip(),
                    "task_type": task_type,
                }
                st.session_state.active_session_id = session_id
                st.session_state.active_session = session
                first_question = (
                    "我会作为苏格拉底式阅读伙伴来提问，而不是直接替你总结。"
                    "请先用 2-3 句话说说：你认为这篇文献的核心研究问题是什么？你是从哪些文本线索判断的？"
                )
                add_message(session_id, "assistant", first_question)
                st.success(f"已创建阅读会话 #{session_id}")

    if not st.session_state.active_session_id:
        st.info("请先在左侧创建一个阅读会话。")
        return

    session = st.session_state.active_session
    st.subheader(f"当前会话 #{session['id']}：{session['literature_title']}")

    api_key_ready = bool(get_config_value("OPENAI_API_KEY"))
    if not api_key_ready:
        st.warning("尚未配置 OPENAI_API_KEY。你仍可填写和保存笔记，但 AI 聊天需要先配置 .env。")

    tab_chat, tab_note = st.tabs(["AI 对话", "批判性阅读笔记"])

    with tab_chat:
        messages = rows_to_dicts(get_messages(session["id"]))
        for message in messages:
            chat_role = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(chat_role):
                st.write(message["content"])

        user_input = st.chat_input("写下你的分析、困惑或对 AI 问题的回答……")
        if user_input:
            add_message(session["id"], "student", user_input)
            with st.chat_message("user"):
                st.write(user_input)

            latest_history = rows_to_dicts(get_messages(session["id"]))
            try:
                with st.chat_message("assistant"):
                    with st.spinner("AI 正在生成追问……"):
                        ai_reply = call_ai(session, latest_history)
                    st.write(ai_reply)
                add_message(session["id"], "assistant", ai_reply)
            except (RuntimeError, OpenAIError) as exc:
                st.error(f"AI 调用失败：{exc}")

    with tab_note:
        existing_note = get_reading_note(session["id"])
        defaults = dict(existing_note) if existing_note else {}
        with st.form("reading_note_form"):
            research_question = st.text_area("1. 研究问题：作者要解决什么问题？为什么重要？", value=defaults.get("research_question", ""), height=110)
            theoretical_framework = st.text_area("2. 理论框架：核心概念/理论如何支撑研究？", value=defaults.get("theoretical_framework", ""), height=110)
            methodology = st.text_area("3. 研究方法：对象、数据和方法是否匹配问题？", value=defaults.get("methodology", ""), height=110)
            evidence_chain = st.text_area("4. 证据链：结论由哪些证据支持？是否充分？", value=defaults.get("evidence_chain", ""), height=110)
            limitations_contributions = st.text_area("5. 局限与贡献：你如何评价其价值与不足？", value=defaults.get("limitations_contributions", ""), height=110)
            transfer_reflection = st.text_area("6. 迁移反思：对你的研究/教学/实践有何启发？", value=defaults.get("transfer_reflection", ""), height=110)
            open_questions = st.text_area("7. 尚未解决的问题：还想继续追问什么？", value=defaults.get("open_questions", ""), height=90)
            note_submitted = st.form_submit_button("保存阅读笔记")

        if note_submitted:
            upsert_reading_note(
                session["id"],
                session["student_id"],
                session["week_node"],
                session["literature_title"],
                research_question,
                theoretical_framework,
                methodology,
                evidence_chain,
                limitations_contributions,
                transfer_reflection,
                open_questions,
            )
            st.success("阅读笔记已保存。")


def render_teacher_page() -> None:
    """Teacher workflow: inspect notes/logs and export CSV files."""
    st.header("教师端：阅读过程与批判性笔记管理")

    with st.sidebar:
        if st.button("返回角色选择", use_container_width=True):
            switch_role(None)
        st.divider()
        st.markdown("**教师管理区**")
        st.caption("查看阅读过程、笔记与导出数据。")

    sessions = rows_to_dicts(get_sessions())
    messages_df = get_messages_dataframe()
    notes_df = get_reading_notes_dataframe()

    st.markdown("<div class=\"section-label\">学生阅读会话概览</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("阅读会话数", len(sessions))
    col2.metric("对话消息数", len(messages_df))
    col3.metric("阅读笔记数", len(notes_df))

    st.markdown("<div class=\"section-label\" style=\"margin-top: 1.2rem;\">数据导出或查看区域</div>", unsafe_allow_html=True)
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "导出对话日志 CSV",
            dataframe_to_csv_bytes(messages_df),
            file_name="dialogue_logs.csv",
            mime="text/csv",
            disabled=messages_df.empty,
        )
    with export_col2:
        st.download_button(
            "导出阅读笔记 CSV",
            dataframe_to_csv_bytes(notes_df),
            file_name="reading_notes.csv",
            mime="text/csv",
            disabled=notes_df.empty,
        )

    tab_overview, tab_notes, tab_logs = st.tabs(["文献阅读记录", "学生批判性笔记", "对话日志"])
    with tab_overview:
        if sessions:
            st.dataframe(sessions, use_container_width=True, hide_index=True)
        else:
            st.info("暂无阅读会话记录。")

    tab_notes, tab_logs = tab_notes, tab_logs
    with tab_notes:
        st.dataframe(notes_df, use_container_width=True, hide_index=True)

    with tab_logs:
        if not sessions:
            st.info("暂无会话数据。")
            return
        options = {
            f"#{s['id']} | {s['student_id']} | {s['week_node']} | {s['literature_title']}": s["id"]
            for s in sessions
        }
        selected = st.selectbox("选择一个学生会话查看对话", list(options.keys()))
        session_id = options[selected]
        session_messages = rows_to_dicts(get_messages(session_id))
        for message in session_messages:
            role_label = "AI" if message["role"] == "assistant" else "学生"
            st.markdown(f"**{role_label}**  ")
            st.write(message["content"])
            st.caption(message["created_at"])
            st.divider()


def main() -> None:
    ensure_session_state()
    inject_global_css()

    role = st.session_state.get("role")
    if role is None:
        render_role_selection()
    elif role == "student":
        render_student_page()
    elif role == "teacher":
        render_teacher_page()
    else:
        st.session_state["role"] = None
        render_role_selection()

if __name__ == "__main__":
    main()

