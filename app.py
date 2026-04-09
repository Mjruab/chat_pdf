import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Insight PDF Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS (MISMO SISTEMA)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #fffde7; color: #333; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fff9c4 !important;
    border-right: 1px solid #f9a825;
}

/* Headers */
h1 { color: #f57f17 !important; }
h2, h3 { color: #e65100 !important; }

/* Inputs */
textarea, input[type="text"] {
    background-color: #fffff0 !important;
    border: 1px solid #f9a825 !important;
    border-radius: 6px !important;
}

/* Botones */
.stButton > button {
    background: #f9a825 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    width: 100%;
}
.stButton > button:hover {
    background: #f57f17 !important;
}

/* Cards */
.header-card {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    padding: 28px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.section-card {
    background: #fff8e1;
    border: 1px solid #ffe082;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Insight PDF Agent")
    st.divider()
    st.markdown("Analiza documentos PDF con IA")
    st.markdown("### Configuración")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>📄 Insight PDF Agent</h1>
    <p>Explora, analiza y consulta documentos PDF mediante IA</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS PRINCIPALES
# ─────────────────────────────────────────────
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown("### 🔑 Clave de OpenAI")
    ke = st.text_input("Ingresa tu API Key", type="password")

    if ke:
        os.environ['OPENAI_API_KEY'] = ke
    else:
        st.warning("Ingresa tu API Key para continuar")

    st.markdown("### 📂 Cargar PDF")
    pdf = st.file_uploader("Sube tu documento", type="pdf")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown("### ℹ️ Información")
    st.write("Versión Python:", platform.python_version())

    try:
        image = Image.open('Chat_pdf.png')
        st.image(image)
    except:
        st.info("Imagen no disponible")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
if pdf is not None and ke:
    try:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown("### ⚙️ Procesamiento del documento")

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"📊 {len(text):,} caracteres extraídos")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20
        )
        chunks = text_splitter.split_text(text)

        st.success(f"Documento dividido en {len(chunks)} partes")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown('</div>', unsafe_allow_html=True)

        # ────────────────
        # PREGUNTAS
        # ────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown("### 💬 Haz una pregunta")
        user_question = st.text_area("",
            placeholder="Ej: ¿Cuál es la idea principal del documento?")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")

            response = chain.run(
                input_documents=docs,
                question=user_question
            )

            st.markdown("### 🧠 Respuesta")
            st.markdown(response)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif pdf is not None and not ke:
    st.warning("Debes ingresar tu API Key")
else:
    st.info("Sube un PDF para comenzar")
