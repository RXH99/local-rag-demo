import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- 更新点 1：从 langchain_ollama 导入 ---
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 加载 data 文件夹下所有 .txt 文档
docs = []
data_dir = "data"
for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(data_dir, file)
        loader = TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load())
        print(f"已加载: {file}")

print(f"\n共加载 {len(docs)} 个文档原始对象")

# 2. 切片：每个 chunk 500 字符，重叠 50 字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"分割后共 {len(chunks)} 个文本片段")

# --- 更新点 2：使用新导入的 OllamaEmbeddings 类 ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. 存入 Chroma 向量数据库（持久化到 ./chroma_db）
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
# --- 更新点 3：删除手动保存的代码行，现在由框架自动处理持久化 ---
print("向量库构建完成，已保存到 ./chroma_db")