from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 1. 检查向量库中是否存在 rag_intro.txt 的片段
all_data = vectordb.get()
sources = set([meta.get('source') for meta in all_data['metadatas']])
print("向量库中包含的源文件：")
for src in sources:
    print(f"  - {src}")

# 2. 检索 "RAG" 相关的前3个片段
print("\n检索 'RAG' 的前3个结果：")
results = vectordb.similarity_search("RAG", k=3)
for idx, doc in enumerate(results):
    print(f"{idx+1}. 来源: {doc.metadata.get('source')}")
    print(f"   内容预览: {doc.page_content[:100]}...")
    print()

# 3. 特别检查 rag_intro.txt 的内容是否被正确编码
if "data\\rag_intro.txt" in sources or "data/rag_intro.txt" in sources:
    rag_docs = vectordb.get(where={"source": "data\\rag_intro.txt"})
    print(f"rag_intro.txt 共有 {len(rag_docs['ids'])} 个片段")
    if rag_docs['ids']:
        print("第一个片段内容：")
        print(rag_docs['documents'][0][:200])
    else:
        print("rag_intro.txt 无片段？")
else:
    print("警告：rag_intro.txt 不在向量库中！")