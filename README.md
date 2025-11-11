# 🦙 LLaMA/DeepSeek 个人知识库问答系统

基于开源LLM模型（LLaMA/DeepSeek）和RAG技术构建的个人知识库问答系统，支持本地文档的智能问答。

## ✨ 功能特性

- 📚 **多格式文档支持**：支持PDF、TXT等格式文档
- 🔍 **智能检索**：基于向量相似度的语义检索
- 🤖 **多模型支持**：兼容LLaMA、DeepSeek等开源模型
- 💬 **友好界面**：基于Gradio的Web交互界面
- 🚀 **本地部署**：完全本地运行，保护数据隐私

## 📋 系统要求

### 硬件要求
- **内存**：至少8GB可用内存
- **存储**：至少5GB可用空间
- **可选GPU**：支持CUDA的NVIDIA显卡（加速推理）

### 软件要求
- **Python** 3.8+
- **pip** 包管理工具

## 🛠️ 安装步骤

### 1. 克隆项目
```bash
git clone <项目地址>
cd llama_kb
```

### 2. 创建虚拟环境（推荐）
```bash
conda create -n llama_kb python=3.10
conda activate llama_kb
```

### 3. 安装依赖
```bash
pip install langchain==0.1.5
pip install langchain-community langchain-core langchain-text-splitters
pip install chromadb gradio sentence-transformers pypdf2
pip install llama-cpp-python
```

### 4. 下载模型
选项A：使用LLaMA模型
从Hugging Face下载GGUF格式的LLaMA模型，例如：

llama-3.2-3b-instruct-q4_0.gguf

放置到项目根目录

选项B：使用DeepSeek模型
bash
# 通过Ollama安装DeepSeek模型
ollama pull deepseek-r1:7b

# 或手动下载GGUF格式的DeepSeek模型

##  快速开始
### 1. 准备知识库文档
在项目根目录创建 docs 文件夹，放入你的文档

### 2. 配置模型路径
编辑 app.py 中的模型路径配置：
MODEL_PATH = "./llama-3.2-3b-instruct-q4_0.gguf"  # 替换为你的模型路径


### 3. 运行系统
```bash
python app.py
```

### 4. 访问系统
在浏览器中打开：http://127.0.0.1:7860


## 使用指南
### 构建知识库
将文档放入 docs 目录

在Web界面点击"🚀 构建/更新知识库"

等待处理完成（显示处理状态）


### 开始问答
在问答界面输入问题

点击"发送"或按Enter键

系统基于知识库内容生成答案

### 支持的文档格式
📄 PDF文档（.pdf）

📝 文本文件（.txt）

🎯 更多格式可通过扩展支持

## 配置说明

### 模型配置
MODEL_PATH = "./your-model.gguf"      # 模型文件路径
DOCS_DIR = "./docs"                   # 文档目录
PERSIST_DIR = "./chroma_db"           # 向量数据库目录

### 模型参数
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,           # 上下文长度
    n_batch=512,          # 批处理大小
    n_gpu_layers=0,       # GPU加速层数（0=仅CPU）
    temperature=0.2,      # 生成随机性
    max_tokens=512,       # 最大生成长度
)

### 检索参数
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}  # 检索文档数量
)

## 高级配置
### GPU加速
如有NVIDIA GPU，可启用GPU加速：

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=35,      # 使用GPU加速（根据显存调整）
    # ... 其他参数
)

### 自定义检索策略
# 最大边际相关性检索（提高多样性）
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)

## 项目结构
llama_kb/    
├── app.py                 # 主应用程序    
├── docs/                  # 文档存储目录    
├── chroma_db/             # 向量数据库    
├── requirements.txt       # 依赖列表    
├── *.gguf                # 模型文件    
└── README.md             # 说明文档    

## 模型替换
### 替换为DeepSeek模型

#### 方法一：使用Ollama
from langchain_community.llms import Ollama

llm = Ollama(model="deepseek-r1:7b")

#### 方法二：使用GGUF格式
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="./deepseek-r1-model.gguf",
    # ... 其他参数
)

## 🎯 性能优化建议
文档预处理：确保文档质量，移除无关内容

分块策略：根据文档类型调整chunk_size（200-1000）

检索优化：根据需求调整检索数量和相似度阈值

硬件利用：合理配置GPU层数，平衡速度与内存使用


