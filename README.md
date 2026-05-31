# Local RAG Demo

基于 **LangChain + Ollama + Chroma** 构建的本地知识库问答系统，完全离线运行。

## ✨ 功能

- 加载本地文本文档（`.txt`），自动切片并生成向量索引
- 使用 Ollama 提供的嵌入模型（`nomic-embed-text`）和大语言模型（`qwen:4b`）
- 检索增强生成（RAG），根据文档内容回答问题
- 支持命令行交互，可选 Streamlit Web 界面

## 🛠️ 技术栈

- Python 3.9+
- LangChain (langchain-classic, langchain-ollama, langchain-chroma)
- Chroma 向量数据库
- Ollama（运行本地 LLM 和 Embedding 模型）

## 🚀 快速开始

### 前置条件

- 安装 Ollama：https://ollama.com
- 拉取模型：

```bash
ollama pull qwen:4b
ollama pull nomic-embed-text
```

## 构建向量库
将你的文本文档放入 data/ 文件夹（已有示例），然后运行：

```bash  
python build_vector_store.py
问答交互
```
## python qa.py
输入问题，基于知识库回答。输入 q 退出。

## Web 界面（可选）
```bash
streamlit run app.py
```
## 📁 项目结构
```
text  文本
.
├── data/                  # 原始文本文档
├── chroma_db/             # 向量库（不提交Git）
├── build_vector_store.py  # 构建向量库脚本
├── qa.py                  # 命令行问答脚本
├── app.py                 # Streamlit 界面（可选）
├── requirements.txt       # 依赖列表
└── README.md
```
### 📝 示例
问题：什么是 RAG？

回答：RAG 是 Retrieval-Augmented Generation 的缩写，中文称为检索增强生成。它是一种结合信息检索和大语言模型的技术，先根据用户问题检索相关文档，再将文档作为上下文送给模型生成答案，从而提高答案的准确性和事实性。

## 🧠 改进方向
支持更多文档格式（PDF、Markdown、Word）

优化 chunk 大小与重叠

增加相似度阈值过滤低质量检索结果

添加对话历史记忆
