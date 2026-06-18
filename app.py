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
    get_comparison_note,
    get_messages,
    get_reading_note,
    get_sessions,
    get_student_feedback,
    get_student_reflection,
    get_teacher_feedback,
    init_db,
    rows_to_dicts,
    upsert_comparison_note,
    upsert_reading_note,
    upsert_student_feedback,
    upsert_student_reflection,
    upsert_teacher_feedback,
)
from export_utils import (
    dataframe_to_csv_bytes,
    get_comparison_notes_dataframe,
    get_messages_dataframe,
    get_reading_notes_dataframe,
    get_student_feedback_dataframe,
    get_student_reflections_dataframe,
    get_teacher_feedback_dataframe,
)
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

BG_COLOR = "#FBF7F1"
SURFACE_COLOR = "#FFFFFF"
TEXT_COLOR = "#221D16"
TEXT_MUTED_COLOR = "#6E665A"
TEXT_FAINT_COLOR = "#A89C88"
BORDER_COLOR = "#ECE5D9"
BORDER_STRONG_COLOR = "#DDC4A8"
ACCENT_COLOR = "#BC6A30"
ACCENT_DARK_COLOR = "#A2571F"
ACCENT_TINT_COLOR = "#F6EDE3"
ACCENT_TINT_BORDER_COLOR = "#E8D6C1"


def inject_global_css() -> None:
    """Apply the warm sand amber visual system without changing app behavior."""
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Noto Sans SC', sans-serif;
        }}
        .stApp {{
            background: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        .block-container {{
            max-width: 1080px;
            padding-top: 2.2rem;
        }}
        h1 {{
            font-family: 'Noto Serif SC', serif !important;
            font-weight: 700 !important;
            color: {TEXT_COLOR} !important;
            font-size: 2rem !important;
            letter-spacing: .01em;
        }}
        h2 {{
            font-family: 'Noto Serif SC', serif !important;
            font-weight: 700 !important;
            color: {TEXT_COLOR} !important;
            font-size: 1.35rem !important;
        }}
        h3 {{
            font-family: 'Noto Serif SC', serif !important;
            font-weight: 600 !important;
            color: {TEXT_COLOR} !important;
            font-size: 1.05rem !important;
        }}
        p, label, .stMarkdown {{
            color: {TEXT_COLOR};
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTED_COLOR} !important;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: {ACCENT_COLOR};
            color: #fff;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            padding: .55rem 1rem;
            transition: background .15s ease, border-color .15s ease;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: {ACCENT_DARK_COLOR};
            color: #fff;
        }}
        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            box-shadow: 0 0 0 3px {ACCENT_TINT_COLOR};
        }}
        .stButton > button[kind="secondary"],
        .stDownloadButton > button[kind="secondary"] {{
            background: #fff;
            color: {ACCENT_COLOR};
            border: 1px solid {BORDER_STRONG_COLOR};
        }}
        .stButton > button[kind="secondary"]:hover,
        .stDownloadButton > button[kind="secondary"]:hover {{
            background: {ACCENT_TINT_COLOR};
            color: {ACCENT_DARK_COLOR};
            border-color: {BORDER_STRONG_COLOR};
        }}
        section[data-testid="stSidebar"] {{
            background: #fff;
            border-right: 1px solid {BORDER_COLOR};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.4rem;
        }}
        .sb-group {{
            font-size: 11px;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: {TEXT_FAINT_COLOR};
            font-weight: 700;
            margin: 1.1rem 0 .5rem;
            padding-bottom: .4rem;
            border-bottom: 1px solid {BORDER_COLOR};
        }}
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input {{
            background: {BG_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 6px !important;
            color: {TEXT_COLOR} !important;
        }}
        .stTextInput input:focus,
        .stTextArea textarea:focus {{
            border-color: {ACCENT_COLOR} !important;
            box-shadow: 0 0 0 3px {ACCENT_TINT_COLOR} !important;
        }}
        .stRadio [aria-checked="true"] svg {{
            color: {ACCENT_COLOR} !important;
            fill: {ACCENT_COLOR} !important;
        }}
        [data-testid="stMetric"] {{
            background: #fff;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 3px rgba(34,29,22,.05);
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Noto Serif SC', serif;
            font-weight: 700;
            color: {TEXT_COLOR};
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED_COLOR};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: .25rem;
            border-bottom: 1px solid {BORDER_COLOR};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {TEXT_MUTED_COLOR};
            font-weight: 500;
            padding: .5rem .9rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT_COLOR} !important;
            border-bottom: 2px solid {ACCENT_COLOR} !important;
        }}
        [data-testid="stDataFrame"], .stDataFrame {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            overflow: hidden;
        }}
        [data-testid="stAlert"] {{
            background: {ACCENT_TINT_COLOR};
            border: 1px solid {ACCENT_TINT_BORDER_COLOR};
            border-radius: 8px;
            color: #7A4A23;
        }}
        [data-testid="stFileUploader"] section {{
            background: {BG_COLOR};
            border: 1px dashed {BORDER_STRONG_COLOR};
            border-radius: 8px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: #fff;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(34,29,22,.05);
        }}
        .role-hero {{
            max-width: 880px;
            margin: 1.8rem auto 1.3rem auto;
            text-align: center;
        }}
        .role-title {{
            font-family: 'Noto Serif SC', serif;
            font-size: 2rem;
            line-height: 1.25;
            font-weight: 700;
            color: {TEXT_COLOR};
            margin-bottom: .45rem;
        }}
        .role-subtitle {{
            font-size: 1rem;
            color: {TEXT_MUTED_COLOR};
            margin: 0 auto;
            max-width: 760px;
            line-height: 1.75;
        }}
        .role-card-copy {{
            min-height: 132px;
        }}
        .role-card-title {{
            font-family: 'Noto Serif SC', serif;
            font-size: 1.22rem;
            font-weight: 700;
            color: {TEXT_COLOR};
            margin-bottom: .55rem;
        }}
        .role-card-desc {{
            color: {TEXT_MUTED_COLOR};
            line-height: 1.75;
            font-size: .96rem;
        }}
        .section-label {{
            color: {ACCENT_COLOR};
            font-weight: 700;
            font-size: .86rem;
            letter-spacing: .08em;
            margin: 1rem 0 .35rem;
        }}
        .muted-copy {{
            color: {TEXT_MUTED_COLOR};
        }}
        [data-testid="stChatMessage"] {{
            background: #fff;
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(34,29,22,.04);
            padding: .75rem .9rem;
        }}
        [data-testid="stChatMessageAvatar"] {{
            background: {ACCENT_COLOR};
            color: white;
            font-weight: 700;
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
            if st.button("进入学生端", use_container_width=True, key="enter_student", type="primary"):
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
            if st.button("进入教师端", use_container_width=True, key="enter_teacher", type="secondary"):
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



def render_comparison_tab(session: dict) -> None:
    """Render and save the T2 double-literature comparison table."""
    existing = get_comparison_note(session["id"])
    defaults = dict(existing) if existing else {}
    with st.form("comparison_note_form"):
        st.markdown("<div class=\"section-label\">T2 双文献比较表</div>", unsafe_allow_html=True)
        comparison_literature_title = st.text_input(
            "第二篇对比文献题目",
            value=defaults.get("comparison_literature_title", ""),
            placeholder="请输入用于比较的另一篇文献题目",
        )
        research_question_comparison = st.text_area(
            "1. 研究问题比较：两篇文献分别解决什么问题？问题意识有何差异？",
            value=defaults.get("research_question_comparison", ""),
            height=100,
        )
        theory_comparison = st.text_area(
            "2. 理论/概念框架比较：两篇文献使用的理论是否不同？适切性如何？",
            value=defaults.get("theory_comparison", ""),
            height=100,
        )
        method_comparison = st.text_area(
            "3. 方法比较：研究对象、数据来源、分析方法有什么差异？哪种更适合回答问题？",
            value=defaults.get("method_comparison", ""),
            height=100,
        )
        evidence_comparison = st.text_area(
            "4. 证据链比较：两篇文献的证据是否充分？结论是否存在过度推断？",
            value=defaults.get("evidence_comparison", ""),
            height=100,
        )
        contribution_limit_comparison = st.text_area(
            "5. 贡献与局限比较：两篇文献各自贡献、局限和改进方向是什么？",
            value=defaults.get("contribution_limit_comparison", ""),
            height=100,
        )
        synthesis_reflection = st.text_area(
            "6. 综合反思：比较后，你对该主题或自己的研究问题有什么新理解？",
            value=defaults.get("synthesis_reflection", ""),
            height=100,
        )
        submitted = st.form_submit_button("保存双文献比较表")

    if submitted:
        upsert_comparison_note(
            session["id"],
            session["student_id"],
            session["week_node"],
            session["literature_title"],
            comparison_literature_title,
            research_question_comparison,
            theory_comparison,
            method_comparison,
            evidence_comparison,
            contribution_limit_comparison,
            synthesis_reflection,
        )
        st.success("双文献比较表已保存。")


def render_reflection_feedback_tab(session: dict) -> None:
    """Render and save student reflection text plus short feedback questionnaire."""
    existing_reflection = get_student_reflection(session["id"])
    reflection_defaults = dict(existing_reflection) if existing_reflection else {}
    existing_feedback = get_student_feedback(session["id"])
    feedback_defaults = dict(existing_feedback) if existing_feedback else {}

    with st.form("student_reflection_form"):
        st.markdown("<div class=\"section-label\">学生反思文本</div>", unsafe_allow_html=True)
        reflection_stage = st.selectbox(
            "反思阶段",
            ["T1 单篇深读后", "T2 双文献比较后", "T3 文献汇报后", "后测/总结反思"],
            index=["T1 单篇深读后", "T2 双文献比较后", "T3 文献汇报后", "后测/总结反思"].index(reflection_defaults.get("reflection_stage", "T1 单篇深读后"))
            if reflection_defaults.get("reflection_stage") in ["T1 单篇深读后", "T2 双文献比较后", "T3 文献汇报后", "后测/总结反思"]
            else 0,
        )
        evidence_use_reflection = st.text_area(
            "1. 我是否回到原文寻找证据？哪些回答是基于原文线索作出的？",
            value=reflection_defaults.get("evidence_use_reflection", ""),
            height=100,
        )
        revised_understanding = st.text_area(
            "2. 在 AI 追问后，我修正了哪些原有理解？",
            value=reflection_defaults.get("revised_understanding", ""),
            height=100,
        )
        ai_dependency_reflection = st.text_area(
            "3. 我是否依赖 AI 摘要或判断？我如何核查 AI 的提示？",
            value=reflection_defaults.get("ai_dependency_reflection", ""),
            height=100,
        )
        remaining_questions = st.text_area(
            "4. 我仍然想追问的问题，或下一步需要重新阅读的部分",
            value=reflection_defaults.get("remaining_questions", ""),
            height=90,
        )
        reflection_submitted = st.form_submit_button("保存反思文本")

    if reflection_submitted:
        upsert_student_reflection(
            session["id"],
            session["student_id"],
            session["week_node"],
            session["literature_title"],
            reflection_stage,
            evidence_use_reflection,
            revised_understanding,
            ai_dependency_reflection,
            remaining_questions,
        )
        st.success("学生反思文本已保存。")

    with st.form("student_feedback_form"):
        st.markdown("<div class=\"section-label\">即时/阶段反馈问卷</div>", unsafe_allow_html=True)
        st.caption("1 = 非常不同意 / 很低，5 = 非常同意 / 很高")
        usefulness = st.slider("AI 支架对我理解文献有帮助", 1, 5, int(feedback_defaults.get("usefulness", 3) or 3))
        ease_of_use = st.slider("这个工具容易使用", 1, 5, int(feedback_defaults.get("ease_of_use", 3) or 3))
        critical_reading_support = st.slider("AI 追问促进了我的批判性阅读", 1, 5, int(feedback_defaults.get("critical_reading_support", 3) or 3))
        ai_dependency_awareness = st.slider("这个过程降低了我对 AI 摘要的直接依赖", 1, 5, int(feedback_defaults.get("ai_dependency_awareness", 3) or 3))
        satisfaction = st.slider("整体使用满意度", 1, 5, int(feedback_defaults.get("satisfaction", 3) or 3))
        helpful_questions = st.text_area("哪些 AI 追问最有帮助？", value=feedback_defaults.get("helpful_questions", ""), height=90)
        improvement_suggestions = st.text_area("你希望下一轮如何改进问题链或模板？", value=feedback_defaults.get("improvement_suggestions", ""), height=90)
        feedback_submitted = st.form_submit_button("保存反馈问卷")

    if feedback_submitted:
        upsert_student_feedback(
            session["id"],
            session["student_id"],
            session["week_node"],
            session["literature_title"],
            usefulness,
            ease_of_use,
            critical_reading_support,
            ai_dependency_awareness,
            satisfaction,
            helpful_questions,
            improvement_suggestions,
        )
        st.success("学生反馈问卷已保存。")
def render_student_page() -> None:
    """Student workflow: create reading session, chat, and submit reading notes."""
    st.header("学生端：AI 苏格拉底式文献阅读支架")
    st.caption("V1.0 原型：用于研究生课程文献阅读节点，保存对话与批判性阅读笔记。")

    with st.sidebar:
        if st.button("返回角色选择", use_container_width=True, type="secondary"):
            switch_role(None)
        st.divider()
        st.markdown('<div class="sb-group">阅读会话设置</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="sb-group">文档信息</div>', unsafe_allow_html=True)
            student_id = st.text_input("学生编号", placeholder="如 S001")
            week_node = st.selectbox("周次节点", ["T0", "T1", "T2", "T3"])
            literature_title = st.text_input("文献题目")
            if input_mode == "手动复制粘贴":
                literature_excerpt = st.text_area("文献摘要或正文节选", height=160)
            else:
                literature_excerpt = st.session_state.uploaded_literature_text
                if st.session_state.uploaded_literature_name:
                    st.caption(f"当前上传文件：{st.session_state.uploaded_literature_name}")
            st.markdown('<div class="sb-group">任务类型</div>', unsafe_allow_html=True)
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

    tab_chat, tab_note, tab_compare, tab_reflection = st.tabs(["AI 对话", "批判性阅读笔记", "双文献比较", "反思与反馈"])

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



    with tab_compare:
        render_comparison_tab(session)

    with tab_reflection:
        render_reflection_feedback_tab(session)

def render_teacher_page() -> None:
    """Teacher workflow: inspect notes/logs and export CSV files."""
    st.header("教师端：阅读过程与批判性笔记管理")

    with st.sidebar:
        if st.button("返回角色选择", use_container_width=True, type="secondary"):
            switch_role(None)
        st.divider()
        st.markdown('<div class="sb-group">教师管理区</div>', unsafe_allow_html=True)
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
            type="primary",
        )
    with export_col2:
        st.download_button(
            "导出阅读笔记 CSV",
            dataframe_to_csv_bytes(notes_df),
            file_name="reading_notes.csv",
            mime="text/csv",
            disabled=notes_df.empty,
            type="secondary",
        )

    tab_overview, tab_notes, tab_research, tab_teacher_feedback, tab_logs = st.tabs(["文献阅读记录", "学生批判性笔记", "DBR研究数据", "教师反馈评分", "对话日志"])
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

