%%writefile app.py
import streamlit as st
import io
from huggingface_hub import notebook_login
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from transformers import BitsAndBytesConfig, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA


st.set_page_config(page_title="Financial Analysis Chatbot", page_icon="💼")

st.markdown("""
    <style>
    /* CSS styles */
    </style>
""", unsafe_allow_html=True)

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

def read_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Hugging Face Login")
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    submit_button = st.button("Login")

    if submit_button:
        try:
            notebook_login(hf_token)
            st.success("Connected successfully to Hugging Face Hub.")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.subheader("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
    process_button = st.button("Process PDFs")

st.header("Financial Analysis Chatbot 💼")

if process_button and uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        text = read_pdf(io.BytesIO(bytes_data))
        documents.append(Document(text))

    st.session_state.documents_processed = True

    combined_text = " ".join([doc.page_content for doc in documents])
    DEVICE = "cpu"
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_text = text_splitter.split_documents(documents)
    

    #intergrate with Chroma databases.
    
    db = Chroma.from_documents(split_text, embeddings, persist_directory="db")

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    # model = CTransformers(model=model_id, max_new_tokens=50, model_file="llama-2-7b-chat.Q5_K_S.gguf")
    
    nf4_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,)

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

    llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto")
    
    
    # retriever = db.as_retriever(search_kwargs={'k': 5})
    # st.session_state.chain = ConversationalRetrievalChain.from_llm(llama_pipeline, retriever, return_source_documents=True)
    
    llm = HuggingFacePipeline(pipeline=llama_pipeline, model_kwargs={"temperature": 0})
    
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    #chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,)

if st.session_state.documents_processed:
    user_query = st.text_input("Ask a question about your documents:", key="user_query")
    if st.button("Submit"):
        if st.session_state.chain and user_query:
            result = st.session_state.chain({'question': user_query, 'chat_history': st.session_state.chat_history})
            st.session_state.chat_history.append((user_query, result['answer']))

            st.subheader("Chat History")
            with st.container():
                for query, response in st.session_state.chat_history:
                    st.markdown(f"**You:** {query}")
                    st.markdown(f"**Financial Analyst:** {response}")
        else:
            st.warning("Please process PDFs before asking questions.")
else:
    st.write("Please upload and process PDFs to enable the chat feature.")
