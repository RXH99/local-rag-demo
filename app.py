import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="本地知识库问答", page_icon="📚")
st.title("📚 本地 RAG 知识库问答")

@st.cache_resource
def load_chain():
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)
        template = """你是一个知识库助手，请严格根据以下上下文回答问题。如果上下文中没有相关信息，请直接回答"不知道"。

上下文：{context}

问题：{question}

回答："""
        prompt = ChatPromptTemplate.from_template(template)
        llm = OllamaLLM(model="qwen:4b", temperature=0.1)
        chain = prompt | llm | StrOutputParser()
        return chain, retriever
    except Exception as e:
        st.error(f"加载失败: {e}")
        return None, None

chain, retriever = load_chain()
if chain is None:
    st.stop()

query = st.text_input("请输入你的问题:")
if query:
    with st.spinner("检索并生成中..."):
        # 先检索文档
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        # 调用链
        answer = chain.invoke({"context": context, "question": query})
    st.markdown("### 回答")
    st.write(answer)
    st.markdown("### 参考来源")
    for doc in docs:
        st.write(f"- {doc.metadata.get('source', '未知')}")