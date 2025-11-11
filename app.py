import os
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ==================== 配置区域 ====================
# 模型路径，请修改为你下载的模型文件实际路径
MODEL_PATH = "./llama-3.2-3b-instruct-q4_0.gguf"
# 知识库文档存放目录
DOCS_DIR = "./docs"
# 向量数据库持久化目录
PERSIST_DIR = "./chroma_db"
# ==================== 配置结束 ====================

from langchain.prompts import PromptTemplate

# 自定义提示词
prompt_template = """基于以下上下文信息，请直接回答问题。如果无法从上下文中得到答案，请说不知道。

上下文：
{context}

问题：
{question}

答案："""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# 初始化文本嵌入模型（用于将文本转换为向量）
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 一个轻量且高效的句子嵌入模型
)

# 初始化LLaMA模型
def load_llm():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,           # 上下文长度，允许更长的文档处理
        n_batch=512,          # 批处理大小，提高处理效率
        n_gpu_layers=0,       # 使用GPU的层数，0表示仅用CPU。如有GPU，可设置为35-50加速
        verbose=False,        # 是否打印详细日志
        temperature=0.2,      # 控制生成随机性（0-1），值越低答案越确定
        max_tokens=512,       # 生成回答的最大长度
    )
    return llm

# 初始化或加载向量数据库
def init_vectorstore():
    if os.path.exists(PERSIST_DIR):
        # 如果已存在，则直接加载
        print("加载已有的向量数据库...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        # 否则，创建一个空的Chroma数据库
        print("创建新的向量数据库...")
        texts = ["这是你知识库的初始文档。请通过界面添加你自己的文档。"]
        return Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIR)

# 处理上传的文档，构建知识库
def process_documents():
    doc_files = []
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue  # 跳过不支持的文件类型
        doc_files.extend(loader.load())
    
    if not doc_files:
        return "未在 'docs' 目录下找到支持的文档（.pdf 或 .txt）。"
    
    # 将文档切分成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # 每个文本块的大小
        chunk_overlap=50   # 块之间的重叠部分，避免语义断裂
    )
    chunks = text_splitter.split_documents(doc_files)
    
    # 将文本块添加到向量数据库
    global vector_db
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return f"知识库构建成功！处理了 {len(doc_files)} 个文档，共生成 {len(chunks)} 个文本块。"

# 问答函数
def ask_question(question, history):
    if 'vector_db' not in globals():
        return "请先初始化或构建知识库。", history
    
    # 从向量数据库中检索与问题最相关的文档片段
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # 返回最相关的3个片段
    
    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将检索到的内容"填充"到提示词中
        retriever=retriever,
        return_source_documents=True,   # 是否返回引用源文档
        chain_type_kwargs={"prompt": PROMPT}  # 使用自定义提示词
    )
    
    # 执行问答
    result = qa_chain.invoke({"query": question})
    print('result:', result)
    history.append([question, result["result"]])
    return "", history  # 清空输入框，更新历史

# 初始化核心组件
print("正在初始化模型和向量数据库...")
llm = load_llm()
vector_db = init_vectorstore()

# 构建Gradio界面
with gr.Blocks(title="LLaMA个人知识库问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 LLaMA 个人知识库问答系统")
    gr.Markdown("上传你的文档到`docs`目录，然后点击**构建知识库**。完成后就可以开始提问了！")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 知识库管理")
            build_btn = gr.Button("🚀 构建/更新知识库", variant="primary")
            build_output = gr.Textbox(label="构建状态", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("## 问答界面")
            chatbot = gr.Chatbot(label="对话历史", height=400)
            question_input = gr.Textbox(
                label="请输入你的问题",
                placeholder="例如：文档中提到的XX是什么？",
                lines=2
            )
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")
    
    # 绑定按钮事件
    build_btn.click(fn=process_documents, outputs=build_output)
    submit_btn.click(fn=ask_question, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    question_input.submit(fn=ask_question, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
