import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="TechNova AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM TECHNOVA UI
# ============================================================

CSS = """
<style>

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 12% 10%,
            rgba(124, 58, 237, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 88% 82%,
            rgba(14, 165, 233, 0.13),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #040714 0%,
            #081027 48%,
            #070615 100%
        );

    color: #f8fafc;
}

[data-testid="stAppViewContainer"] .main {
    background: transparent;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 1rem !important;
    padding-bottom: 7rem !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.tech-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 5px 20px;
}

.tech-logo {
    width: 56px;
    height: 56px;
    border-radius: 18px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 27px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #4f46e5,
            #06b6d4
        );

    box-shadow:
        0 0 28px rgba(124, 58, 237, 0.45),
        inset 0 0 14px rgba(255, 255, 255, 0.12);
}

.tech-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #f8fafc;
}

.tech-title span {
    background:
        linear-gradient(
            90deg,
            #c4b5fd,
            #60a5fa,
            #38bdf8
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.tech-subtitle {
    margin-top: 3px;
    color: #a7b2cf;
    font-size: 13px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080d24 0%,
            #05091b 100%
        );

    border-right:
        1px solid rgba(139, 92, 246, 0.22);
}

[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

.sidebar-card {
    padding: 20px;
    border-radius: 22px;
    margin-bottom: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(124, 58, 237, 0.20),
            rgba(15, 23, 42, 0.72)
        );

    border:
        1px solid rgba(139, 92, 246, 0.32);

    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.28);
}

.sidebar-icon {
    font-size: 36px;
    margin-bottom: 8px;
}

.sidebar-title {
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
}

.sidebar-text {
    color: #aab5d0;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 5px;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    margin-top: 12px;
    padding: 7px 12px;

    border-radius: 999px;

    background:
        rgba(34, 197, 94, 0.09);

    border:
        1px solid rgba(34, 197, 94, 0.30);

    color: #86efac;

    font-size: 12px;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 6px #22c55e,
        0 0 14px rgba(34, 197, 94, 0.80);
}

.section-label {
    color: #9ca8c7;
    font-size: 11px;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: 1px;

    margin: 22px 0 10px 3px;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 48px;

    border-radius: 15px;

    border:
        1px solid rgba(96, 165, 250, 0.22);

    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 80, 0.82),
            rgba(30, 41, 90, 0.62)
        );

    color: #e8edff;

    font-size: 13px;
    font-weight: 650;

    transition:
        transform 0.18s ease,
        border-color 0.18s ease,
        background 0.18s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);

    border-color:
        rgba(129, 140, 248, 0.75);

    background:
        linear-gradient(
            135deg,
            rgba(91, 33, 182, 0.40),
            rgba(37, 99, 235, 0.28)
        );
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.92) !important;
    color: #f8fafc !important;

    border:
        1px solid rgba(139, 92, 246, 0.30) !important;

    border-radius: 14px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] hr {
    border-color:
        rgba(148, 163, 184, 0.13);
}

.sidebar-footer {
    text-align: center;

    color: #66718f;

    font-size: 11px;
    line-height: 1.5;

    padding:
        24px 4px 30px;
}


/* ============================================================
   WELCOME CARD
   ============================================================ */

.welcome-card {
    padding: 26px 28px;
    margin: 8px 0 22px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.18),
            rgba(14, 165, 233, 0.08)
        );

    border:
        1px solid rgba(139, 92, 246, 0.27);

    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 18px 45px rgba(0, 0, 0, 0.16);
}

.welcome-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 8px;
}

.welcome-text {
    color: #aeb8d2;
    line-height: 1.7;
    font-size: 14px;
}

.welcome-text b {
    color: #dbe4ff;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

[data-testid="stChatMessage"] {
    border-radius: 20px !important;

    margin:
        10px 0 !important;

    padding:
        8px 12px !important;

    background:
        rgba(15, 23, 42, 0.72) !important;

    border:
        1px solid rgba(148, 163, 184, 0.14) !important;

    box-shadow:
        0 12px 32px rgba(0, 0, 0, 0.16);
}

[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"] {
    color: #eef2ff !important;

    font-size: 15px !important;

    line-height: 1.75 !important;
}

[data-testid="stChatMessage"] strong {
    color: #ffffff;
}


/* ============================================================
   SOURCES
   ============================================================ */

.source-box {
    margin:
        -3px 0 15px 55px;

    padding:
        9px 13px;

    border-radius: 12px;

    background:
        rgba(124, 58, 237, 0.08);

    border:
        1px solid rgba(139, 92, 246, 0.20);

    color: #aeb8d2;

    font-size: 12px;
}

.source-box b {
    color: #c4b5fd;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

[data-testid="stChatInput"] {
    position: fixed !important;

    bottom: 1.25rem !important;

    left:
        calc(50% + 125px) !important;

    transform:
        translateX(-50%) !important;

    width:
        min(850px, calc(100vw - 430px)) !important;

    z-index: 999 !important;
}

[data-testid="stChatInput"] > div {
    background:
        linear-gradient(
            135deg,
            rgba(13, 20, 48, 0.98),
            rgba(25, 18, 57, 0.98)
        ) !important;

    border:
        1px solid rgba(139, 92, 246, 0.68) !important;

    border-radius: 19px !important;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.45),
        0 0 25px rgba(124, 58, 237, 0.14) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;

    color: #ffffff !important;

    -webkit-text-fill-color:
        #ffffff !important;

    caret-color:
        #ffffff !important;

    font-size: 15px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #aeb8d2 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #4f46e5
        ) !important;

    color: #ffffff !important;

    border-radius: 14px !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .tech-title {
        font-size: 24px;
    }

    [data-testid="stChatInput"] {
        left: 50% !important;

        width:
            calc(100vw - 32px) !important;
    }

}

</style>
"""

st.html(CSS)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="tech-header">

        <div class="tech-logo">
            ✨
        </div>

        <div>

            <div class="tech-title">
                <span>TechNova</span> AI Assistant
            </div>

            <div class="tech-subtitle">
                Intelligent answers powered by TechNova's
                knowledge base
            </div>

        </div>

    </div>
    """
)


# ============================================================
# GROQ API
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )
    st.stop()


@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        groq_api_key=api_key,
    )


llm = get_llm()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_resource
def load_documents():

    documents_path = Path("documents")

    if not documents_path.exists():
        return []

    loader = DirectoryLoader(
        str(documents_path),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        },
        show_progress=False,
    )

    return loader.load()


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

@st.cache_resource
def create_chunks():

    documents = load_documents()

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    return splitter.split_documents(documents)


chunks = create_chunks()


# ============================================================
# VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_vector_database():

    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=
        "technova_documents",
    )


vectorstore = create_vector_database()


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# CASUAL QUESTION DETECTION
# ============================================================

def detect_casual_intent(text):

    text = text.lower().strip()

    casual_patterns = [

        "hello",
        "hi",
        "hey",
        "helo",
        "hii",

        "good morning",
        "good afternoon",
        "good evening",

        "how are you",
        "how r u",
        "how are u",

        "thanks",
        "thank you",
        "thankyou",
        "thx",

        "bye",
        "goodbye",
        "see you",

        "nice to meet you",

        "who are you",
        "what are you",

        "who built you",
        "who build you",
        "who made you",
        "who created you",

        "your creator",
        "your developer",

        "are you an ai",

    ]

    return any(
        pattern in text
        for pattern in casual_patterns
    )


# ============================================================
# CASUAL RESPONSE
# ============================================================

def get_casual_response(
    question,
    language,
):

    q = question.lower().strip()


    # ========================================================
    # URDU
    # ========================================================

    if language == "Urdu":

        if any(
            x in q
            for x in [
                "hello",
                "hi",
                "hey",
                "helo",
                "hii",
            ]
        ):

            return (
                "السلام علیکم! 👋 میں "
                "**TechNova AI** ہوں۔ "
                "آپ مجھ سے TechNova کے بارے میں "
                "کوئی بھی سوال پوچھ سکتے ہیں۔ 😊"
            )


        if (
            "how are you" in q
            or "how r u" in q
        ):

            return (
                "میں بالکل ٹھیک ہوں، شکریہ! 😊 "
                "میں TechNova کے بارے میں "
                "آپ کی مدد کے لیے تیار ہوں۔"
            )


        if "thank" in q:

            return (
                "خوشی ہوئی! 😊 اگر آپ کو "
                "TechNova کے بارے میں مزید "
                "معلومات چاہییں تو بے جھجھک پوچھیں۔"
            )


        if (
            "bye" in q
            or "goodbye" in q
        ):

            return (
                "خدا حافظ! 👋 جب بھی TechNova کے "
                "بارے میں مدد چاہیے ہو، واپس آئیے گا۔"
            )


        if (
            "who are you" in q
            or "what are you" in q
        ):

            return (
                "میں **TechNova AI** ہوں — "
                "TechNova کے لیے بنایا گیا "
                "ایک intelligent business assistant۔ 🤖"
            )


        if any(
            x in q
            for x in [
                "who built you",
                "who build you",
                "who made you",
                "who created you",
            ]
        ):

            return (
                "میں **TechNova AI** ہوں، ایک AI "
                "assistant جو TechNova کے "
                "knowledge-assistant demo کے لیے "
                "بنایا گیا ہے۔ 🤖✨"
            )


        return (
            "جی بالکل! 😊 میں آپ کی مدد کے لیے حاضر ہوں۔"
        )


    # ========================================================
    # ROMAN URDU
    # ========================================================

    if language == "Roman Urdu":

        if any(
            x in q
            for x in [
                "hello",
                "hi",
                "hey",
                "helo",
                "hii",
            ]
        ):

            return (
                "Assalam-o-Alaikum! 👋 Main "
                "**TechNova AI** hoon. "
                "Aap mujh se TechNova ke baray mein "
                "koi bhi sawal pooch sakte hain. 😊"
            )


        if (
            "how are you" in q
            or "how r u" in q
        ):

            return (
                "Main bilkul theek hoon, shukriya! 😊 "
                "Main TechNova ke baray mein "
                "aapki madad ke liye ready hoon."
            )


        if "thank" in q:

            return (
                "You're most welcome! 😊 "
                "Agar TechNova ke baray mein koi "
                "aur sawal ho to pooch sakte hain."
            )


        if (
            "bye" in q
            or "goodbye" in q
        ):

            return (
                "Allah Hafiz! 👋 Jab bhi TechNova "
                "ke baray mein madad chahiye ho, "
                "wapas aa jaiye ga."
            )


        if (
            "who are you" in q
            or "what are you" in q
        ):

            return (
                "Main **TechNova AI** hoon — "
                "TechNova ke liye banaya gaya "
                "ek intelligent business assistant. 🤖"
            )


        if any(
            x in q
            for x in [
                "who built you",
                "who build you",
                "who made you",
                "who created you",
            ]
        ):

            return (
                "Main **TechNova AI** hoon, ek AI "
                "assistant jo TechNova ke "
                "knowledge-assistant demo ke liye "
                "banaya gaya hai. 🤖✨"
            )


        return (
            "Jee bilkul! 😊 Main aapki madad "
            "ke liye hazir hoon."
        )


    # ========================================================
    # ENGLISH / AUTO
    # ========================================================

    if any(
        x in q
        for x in [
            "hello",
            "hi",
            "hey",
            "helo",
            "hii",
        ]
    ):

        return (
            "Hello! 👋 Welcome to **TechNova AI**. "
            "How can I help you today?"
        )


    if (
        "how are you" in q
        or "how r u" in q
    ):

        return (
            "I'm doing great, thank you! 😊 "
            "I'm here and ready to help with "
            "anything related to TechNova."
        )


    if "thank" in q:

        return (
            "You're most welcome! 😊 "
            "Feel free to ask me anything about TechNova."
        )


    if (
        "bye" in q
        or "goodbye" in q
    ):

        return (
            "Goodbye! 👋 It was a pleasure helping you. "
            "Have a wonderful day!"
        )


    if (
        "who are you" in q
        or "what are you" in q
    ):

        return (
            "I'm **TechNova AI** — an intelligent "
            "business assistant built for the "
            "TechNova knowledge-assistant experience. 🤖"
        )


    if any(
        x in q
        for x in [
            "who built you",
            "who build you",
            "who made you",
            "who created you",
        ]
    ):

        return (
            "I'm **TechNova AI**, an AI assistant "
            "built for the TechNova "
            "knowledge-assistant demo. 🤖✨"
        )


    return (
        "Absolutely! 😊 I'm here to help. "
        "What would you like to know about TechNova?"
    )


# ============================================================
# SMART DOCUMENT ROUTING
# ============================================================

def get_relevant_documents(question):

    q = question.lower()

    target_files = []


    # SERVICES

    if any(
        word in q
        for word in [
            "service",
            "services",
            "offer",
            "offering",
            "solution",
            "solutions",
        ]
    ):

        target_files.append(
            "services.txt"
        )


    # PRICING

    if any(
        word in q
        for word in [
            "price",
            "pricing",
            "cost",
            "package",
            "packages",
            "plan",
            "plans",
        ]
    ):

        target_files.append(
            "pricing.txt"
        )


    # SUPPORT

    if any(
        word in q
        for word in [
            "refund",
            "return",
            "cancel",
            "cancellation",
            "support",
            "contact",
        ]
    ):

        target_files.append(
            "refund_support_policy.txt"
        )


    # COMPANY

    if any(
        word in q
        for word in [
            "technova",
            "company",
            "about",
            "mission",
            "overview",
            "who is technova",
        ]
    ):

        target_files.append(
            "company_overview.txt"
        )


    # ========================================================
    # HIGH-CONFIDENCE ROUTING
    # ========================================================

    if target_files:

        selected = []

        for doc in chunks:

            filename = Path(
                doc.metadata.get(
                    "source",
                    ""
                )
            ).name

            if filename in target_files:

                selected.append(doc)


        if selected:
            return selected


    # ========================================================
    # SEMANTIC FALLBACK
    # ========================================================

    if vectorstore is not None:

        try:

            results = (
                vectorstore
                .similarity_search_with_relevance_scores(
                    question,
                    k=6,
                )
            )

            return [
                doc
                for doc, score in results
                if score >= 0.20
            ]

        except Exception:

            return (
                vectorstore
                .similarity_search(
                    question,
                    k=5,
                )
            )


    return []


# ============================================================
# SIDEBAR
# ============================================================

selected_question = None


with st.sidebar:

    st.html(
        """
        <div class="sidebar-card">

            <div class="sidebar-icon">
                🤖
            </div>

            <div class="sidebar-title">
                TechNova AI
            </div>

            <div class="sidebar-text">
                Your intelligent business assistant
                for TechNova's products, services,
                pricing, and company information.
            </div>

            <div class="status">

                <span class="status-dot"></span>

                AI Assistant Online

            </div>

        </div>
        """
    )


    st.html(
        '<div class="section-label">'
        'Quick Questions'
        '</div>'
    )


    quick_questions = [

        (
            "🚀",
            "What services do you offer?"
        ),

        (
            "💰",
            "What are your pricing plans?"
        ),

        (
            "🏢",
            "Tell me about TechNova."
        ),

        (
            "🤖",
            "What AI solutions do you provide?"
        ),

    ]


    for icon, question_text in quick_questions:

        if st.button(
            f"{icon}  {question_text}",
            key=f"quick_{question_text}",
            use_container_width=True,
        ):

            selected_question = question_text


    st.html(
        '<div class="section-label">'
        'Response Language'
        '</div>'
    )


    language = st.selectbox(
        "Language",
        [
            "Auto",
            "English",
            "Urdu",
            "Roman Urdu",
        ],
        label_visibility="collapsed",
    )


    st.markdown("---")


    if st.button(
        "🧹  Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.html(
        """
        <div class="sidebar-footer">

            🔒 Responses are grounded in
            TechNova's knowledge base.

            <br>

            Built for a professional
            RAG demonstration.

        </div>
        """
    )


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.html(
        """
        <div class="welcome-card">

            <div class="welcome-title">
                👋 Welcome to TechNova AI
            </div>

            <div class="welcome-text">

                Ask questions about TechNova's
                <b>
                services, AI solutions, pricing,
                support,
                </b>
                and company information.

                I'll provide reliable answers
                based on the available knowledge base.

            </div>

        </div>
        """
    )


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    avatar = (
        "👤"
        if message["role"] == "user"
        else "✨"
    )


    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )


    if (
        message["role"] == "assistant"
        and message.get("sources")
    ):

        source_text = " • ".join(
            sorted(
                set(
                    message["sources"]
                )
            )
        )


        st.html(
            f"""
            <div class="source-box">

                📚 <b>Sources:</b>
                {source_text}

            </div>
            """
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask TechNova AI anything..."
)


if selected_question:

    question = selected_question


# ============================================================
# QUESTION HANDLER
# ============================================================

if question:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # ========================================================
    # QUICK QUESTION = ALWAYS USE RAG
    # ========================================================

    is_quick_question = (
        selected_question is not None
    )

    # ========================================================
    # CASUAL CHAT
    # ONLY FOR NORMAL USER MESSAGES
    # ========================================================

    if (
        detect_casual_intent(question)
        and not is_quick_question
    ):

        answer = get_casual_response(
            question,
            language,
        )

        source_names = []


    # ========================================================
    # NO VECTOR DATABASE
    # ========================================================

    elif vectorstore is None:

        answer = (
            "I'm sorry, but the TechNova "
            "knowledge base is currently unavailable."
        )

        source_names = []


    # ========================================================
    # RAG
    # ========================================================

    else:

        with st.spinner(
            "🔎 Searching TechNova knowledge..."
        ):

            retrieved_docs = (
                get_relevant_documents(
                    question
                )
            )


            context_parts = []

            source_names = []


            for doc in retrieved_docs:

                text = (
                    doc.page_content.strip()
                )


                if text:

                    context_parts.append(
                        text
                    )


                source = doc.metadata.get(
                    "source"
                )


                if source:

                    filename = Path(
                        source
                    ).name


                    if filename not in source_names:

                        source_names.append(
                            filename
                        )


            # =================================================
            # NO RELEVANT INFORMATION
            # =================================================

            if not context_parts:

                answer = (
                    "I don't have that information "
                    "in my current TechNova "
                    "knowledge base."
                )

                source_names = []


            # =================================================
            # LLM
            # =================================================

            else:

                context = (
                    "\n\n---\n\n"
                    .join(
                        context_parts
                    )
                )


                # =============================================
                # LANGUAGE
                # =============================================

                if language == "Urdu":

                    language_instruction = (
                        "Respond completely in Urdu script."
                    )

                elif language == "Roman Urdu":

                    language_instruction = (
                        "Respond naturally in Roman Urdu."
                    )

                elif language == "English":

                    language_instruction = (
                        "Respond in clear professional English."
                    )

                else:

                    language_instruction = (
                        "Respond naturally in the same "
                        "language used by the user."
                    )


                # =============================================
                # CONVERSATION HISTORY
                # =============================================

                history_text = ""


                for old_message in (
                    st.session_state.messages[-10:]
                ):

                    history_text += (
                        f'{old_message["role"].upper()}: '
                        f'{old_message["content"]}\n'
                    )


                # =============================================
                # PROFESSIONAL RAG PROMPT
                # =============================================

                prompt = f"""
You are TechNova AI Assistant.

You are a professional, friendly,
reliable customer-facing AI assistant.

IMPORTANT RULES:

1. For TechNova-specific questions,
use ONLY the provided TechNova
knowledge base.

2. Never invent TechNova information,
prices, policies, services, people,
or company facts.

3. Answer the user's actual question,
not simply the retrieved text.

4. If the knowledge base does not
contain enough information for a
TechNova-specific question, say:

"I don't have that information in my
current TechNova knowledge base."

5. Never make up a source.

6. Never claim that a document contains
information that it does not contain.

7. Use natural, professional language.

8. Use Markdown formatting.

9. Use **bold** for important information.

10. Use bullet points when helpful.

11. Keep answers concise but useful.

12. Understand follow-up questions
using conversation history.

13. Do not mention internal RAG
processes unless the user asks.

14. {language_instruction}


CONVERSATION HISTORY:

{history_text}


TECHNOVA KNOWLEDGE BASE:

{context}


USER QUESTION:

{question}


Provide the best answer supported
by the TechNova knowledge base.
"""


                response = llm.invoke(
                    prompt
                )


                answer = str(
                    response.content
                ).strip()


                # =============================================
                # UNSUPPORTED ANSWER = NO SOURCE
                # =============================================

                if (
                    "I don't have that information "
                    "in my current TechNova "
                    "knowledge base."
                    in answer
                ):

                    source_names = []


    # ========================================================
    # CURRENT USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(
            question
        )


    # ========================================================
    # CURRENT AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="✨",
    ):

        st.markdown(
            answer
        )


    # ========================================================
    # SOURCE BADGE
    # ONLY WHEN APPROPRIATE
    # ========================================================

    if source_names:

        source_text = " • ".join(
            sorted(
                set(
                    source_names
                )
            )
        )


        st.html(
            f"""
            <div class="source-box">

                📚 <b>Sources:</b>
                {source_text}

            </div>
            """
        )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_names,
        }
    )
