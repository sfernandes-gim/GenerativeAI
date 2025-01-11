import os

from langchain_community.llms import Ollama
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter # Test Semantic Chunker
from langchain.chains import RetrievalQA

working_dir = os.path.dirname(os.path.abspath(__file__))

llm = Ollama(
    model = "llama3.1",
    temperature =0
)

embeddings = HuggingFaceEmbeddings()

def read_pdf(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        content_list = []

        # Iterate through each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            content = page.extract_text()
            content_list.append(content)

    return content_list

def answer_question(file_name, user_query):
    file_path = f"{working_dir}/{file_name}"

    print(f"File path: {file_path}")
   
    content_list = read_pdf(file_path)
    content_list = "\n".join(content_list)

    #Now create the text chunks
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1000, chunk_overlap = 200)
    
    print("Split Text")
    chunks = text_splitter.split_text(content_list)
    print("Chunks Done")
    # Create the Vectire embeddings
    knowledge_base=FAISS.from_texts(chunks, embeddings)
    print("Vector Store Done")
    qa_chain = RetrievalQA.from_chain_type( llm, retriever = knowledge_base.as_retriever())
    print("QA Chain Done")
    response = qa_chain.invoke({"query": user_query})

    return response["result"]

