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
    page_title="RAG PDF App",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# ─────────────────────────────────────────────
# IMAGEN
# ─────────────────────────────────────────────
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=300)
except Exception as e:
    st.warning("No se pudo cargar la imagen")

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("Este Agente analiza PDFs")
    st.markdown("---")

# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Ingresa tu API Key para continuar")

st.markdown("---")

# ─────────────────────────────────────────────
# UPLOADER
# ─────────────────────────────────────────────
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
if pdf is not None and ke:
    try:
        st.info("Procesando PDF...")

        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            contenido = page.extract_text()
            if contenido:  # 🔥 evita error cuando es None
                text += contenido

        if not text.strip():
            st.error("No se pudo extraer texto del PDF")
            st.stop()

        st.success(f"Texto extraído: {len(text):,} caracteres")

        # ─────────────────────────
        # SPLIT
        # ─────────────────────────
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # ─────────────────────────
        # EMBEDDINGS
        # ─────────────────────────
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown("---")

        # ─────────────────────────
        # PREGUNTA
        # ─────────────────────────
        st.subheader("Haz una pregunta sobre el documento")

        user_question = st.text_area(
            "",
            placeholder="Ej: ¿De qué trata el documento?"
        )

        if user_question:
            with st.spinner("Buscando respuesta..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o"
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

                st.markdown("### Respuesta:")
                st.write(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")

elif pdf is not None and not ke:
    st.warning("Debes ingresar tu API Key")

else:
    st.info("Carga un PDF para comenzar")
