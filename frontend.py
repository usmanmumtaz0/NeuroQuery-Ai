import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")  # read key from .env

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=api_key  # pass the key here
)
parser = StrOutputParser()

# ------------------- PROMPTS -------------------
news_prompt = ChatPromptTemplate.from_template(
"You are a professional NEWS expert.\n"
"Start your answer with: 'As a News Expert, '\n"
"Then give a clear and concise response.\n\nQuestion: {input}"
)

fin_prompt = ChatPromptTemplate.from_template(
"You are a professional FINANCE expert.\n"
"Start your answer with: 'As a Finance Expert, '\n"
"Then give a clear and concise response.\n\nQuestion: {input}"
)

tech_prompt = ChatPromptTemplate.from_template(
"You are a professional TECHNOLOGY expert.\n"
"Start your answer with: 'As a Tech Expert, '\n"
"Then give a clear and concise response.\n\nQuestion: {input}"
)


news_chain = news_prompt | llm | parser
fin_chain = fin_prompt | llm | parser
tech_chain = tech_prompt | llm | parser

# ------------------- SMART ROUTER -------------------
def route(q):
    ql = q.lower()

    finance_keywords = ["stock","money","finance","investment","market","bank","economy","crypto","interest","rate","profit","loss"]
    tech_keywords = ["ai","technology","software","hardware","programming","gpt","robot"]
    news_keywords = ["news","breaking","report","update","headline","world","current","politics","war"]

    # 🔥 NEW: detect calculation queries
    math_keywords = ["calculate","find","solve","equation","percentage","percent","interest"]

    has_numbers = any(char.isdigit() for char in q)

    f = sum(word in ql for word in finance_keywords)
    t = sum(word in ql for word in tech_keywords)
    n = sum(word in ql for word in news_keywords)
    m = sum(word in ql for word in math_keywords)

    # ✅ PRIORITY: finance calculation
    if (m > 0 and has_numbers) or f > t and f > n:
        return fin_chain

    elif t > f and t > n:
        return tech_chain

    elif n > f and n > t:
        return news_chain

    # fallback
    return news_chain
# ------------------- PAGE -------------------
st.set_page_config(page_title="NeuroQuery AI", page_icon="🧠", layout="wide")

# ------------------- MODERN CSS -------------------
st.markdown("""
<style>
body {background-color:#0b0f19;}
.block-container {padding-top:1rem;font-family:'Segoe UI';}

.chat-bubble {
    padding:12px 16px;
    border-radius:15px;
    margin:6px 0;
    max-width:75%;
    animation: fadeIn 0.3s ease-in-out;
}

.user {
    background:#1e293b;
    color:white;
    margin-left:auto;
}

.assistant {
    background:#111827;
    color:#00ffd5;
    margin-right:auto;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}

.header {
    text-align:center;
    margin-bottom:20px;
}

<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 70px;
}

.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 10px 0;
    background-color: #0b0f19;
    border-top: 1px solid #333;
    z-index: 999;
}
</style>
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
logo_url = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

st.markdown(f"""
<div class="header">
<img src="{logo_url}" width="110">
<h1 style="background:linear-gradient(90deg,#00ffd5,#00aaff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
NeuroQuery AI
</h1>
<p style="color:gray;">AI Research Assistant • Smart Routing • Gemini Powered</p>
</div>
""", unsafe_allow_html=True)

# ------------------- SESSION -------------------
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "count" not in st.session_state:
    st.session_state.count = 0

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.image(logo_url, width=80)
    st.title("⚙️ Dashboard")

    if st.button("🧹 Clear Chat"):
        st.session_state.clear()
        st.rerun()

    if st.button("📊 Summary"):
        if st.session_state.msgs:
            text = " ".join([m["content"] for m in st.session_state.msgs])
            with st.spinner("Generating summary..."):
                res = llm.invoke("Summarize briefly:\n" + text).content
                summary = res[0] if isinstance(res, list) else str(res)
                st.write(summary)

    st.markdown("---")
    st.write("📈 Total Queries:", st.session_state.count)

# ------------------- CHAT -------------------
st.markdown("### 💬 Chat with NeuroQuery for Tech, News and Finance ")

for m in st.session_state.msgs:
    role_class = "user" if m["role"]=="user" else "assistant"
    st.markdown(f"<div class='chat-bubble {role_class}'>{m['content']}</div>", unsafe_allow_html=True)

q = st.chat_input("Ask anything...")

if q:
    st.session_state.count += 1
    st.session_state.msgs.append({"role":"user","content":q})

    st.markdown(f"<div class='chat-bubble user'>{q}</div>", unsafe_allow_html=True)

    chain = route(q)

    # 🔥 Thinking Spinner (instant feedback)
    with st.spinner("🤖 Thinking..."):
        res = chain.invoke({"input": q})

    # Safe response
    if isinstance(res, dict):
        res_text = res.get("content", "")
    elif isinstance(res, list):
        res_text = str(res[0])
    else:
        res_text = str(res)

    # ⚡ Smooth typing animation (fast like ChatGPT)
    placeholder = st.empty()
    output = ""

    for i in range(0, len(res_text), 3):  # faster chunk typing
        output += res_text[i:i+3]
        placeholder.markdown(f"<div class='chat-bubble assistant'>{output}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.msgs.append({"role":"assistant","content":res_text})

# ------------------- FOOTER -------------------
st.markdown("""
<div class="footer">
© 2026 NeuroQuery AI • Muhammad Usman Mumtaz  🚀
</div>
""", unsafe_allow_html=True)