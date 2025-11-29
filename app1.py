import streamlit as st

# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')

# ---------- RAG CHAIN SETUP (cached so it does NOT reload every time) ----------

@st.cache_resource
def load_rag_chain():
    # Same models you used
    embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.2-nemoretriever-300m-embed-v2")
    llm = ChatNVIDIA(model="openai/gpt-oss-20b")

    # Load your existing FAISS index
    vectorstore = FAISS.load_local(
        folder_path="./nvidia_imdb_faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever()

    prompt_template = ChatPromptTemplate.from_template(
        """
Respond based on provided Context clearly.

### Below is Context you need to use to answer. Consider context as file/document provided to you for answer.                                                   
<context>
{context}
</context>

### Below is User Input
<input>
{input}                                                 
</input>
"""
    )

    # Stuff chain automatically formats documents into {context}
    stuff_chain = create_stuff_documents_chain(llm, prompt_template)

    # Full RAG chain (retriever → stuff → llm)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    return rag_chain


rag_chain = load_rag_chain()

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="IMDB RAG QA", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Movie Q&A (RAG + Ollama)")
st.write("Ask questions about movies based on your IMDB CSV + FAISS index.")

default_question = "Recommend top 5 sci-fi movies with high ratings"

user_query = st.text_input("Ask something:", value=default_question)

col1, col2 = st.columns([1, 3])
with col1:
    ask_button = st.button("Ask")

if ask_button and user_query.strip():
    with st.spinner("Thinking with RAG..."):
        response = rag_chain.invoke({"input": user_query})

    # Main answer
    st.subheader("🧠 Answer")
    st.write(response.get("answer", ""))

    # Show retrieved docs / context
    st.subheader("📚 Retrieved Context")
    context_docs = response.get("context", [])
    if context_docs:
        for i, doc in enumerate(context_docs, start=1):
            with st.expander(f"Context document {i}"):
                st.markdown("**Content:**")
                st.write(doc.page_content)
                if doc.metadata:
                    st.markdown("**Metadata:**")
                    st.json(doc.metadata)
    else:
        st.write("No context documents returned by retriever.")
else:
    st.info("Enter a question and click **Ask** to query your IMDB RAG system.")
