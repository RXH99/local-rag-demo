from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# 1. 加载向量库
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. 检索器
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. 自定义 prompt（带约束：只基于文档，不知道就说不知道）
template = """你是一个知识库助手，请严格根据以下上下文内容回答问题。如果上下文中没有提供相关信息，请不要编造，直接回答“不知道”。

上下文：
{context}

问题：{question}

回答（只基于以上上下文）："""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 4. LLM
llm = OllamaLLM(model="qwen:4b", temperature=0.1)

# 5. 使用自定义 prompt 构建 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",                # 将所有检索到的文档一次性送入 prompt
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=False
)

print("知识库问答系统已就绪！（输入 q 退出）\n")

while True:
    query = input("问题: ")
    if query.lower() == 'q':
        break
    if not query.strip():
        continue

    result = qa_chain.invoke({"query": query})
    print(f"\n回答: {result['result']}\n")
    print("参考文档来源:")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata.get('source', '未知')}")
    print("-" * 50)