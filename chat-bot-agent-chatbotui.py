import gradio as gr
import extract_msg
from dateutil import parser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
import re
import os
import shutil
import pandas as pd
import easyocr

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
llm = Ollama(model="mistral")

def process_pdf(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None, None, None
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    retriever = vectorstore.as_retriever()
    return text_splitter, vectorstore, retriever

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    prompt = "You are an AI Agent which can summarise the output in a point wise manner.\n" + formatted_prompt
    response_content = llm(formatted_prompt)
    if not response_content:
        return "No response from LLM."
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

def process_excel(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No Excel file found."
    try:
        df_dict = pd.read_excel(file_path, sheet_name=None)
        texts = []
        for sheet_name, df in df_dict.items():
            texts.append(f"Sheet: {sheet_name}")
            sheet_text = df.astype(str).agg('\n'.join, axis=1).str.cat(sep='\n')
            texts.append(sheet_text[:2000])
        combined_text = "\n\n".join(texts)
        return ollama_llm("Summarize this Excel data", combined_text)
    except Exception as e:
        return f"Error processing Excel: {str(e)}"

def process_image(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No image file found."
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(file_path, detail=0)
        text = "\n".join(result)
        if not text.strip():
            return "No text detected in the image."
        return ollama_llm("Summarize this extracted text from image", text)
    except Exception as e:
        return f"Error processing image: {str(e)}"

def read_msg_emails_and_summarize(directory):
    emails = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.msg'):
            filepath = os.path.join(directory, filename)
            try:
                msg = extract_msg.Message(filepath)
                msg_sender = msg.sender
                msg_to = msg.to
                msg_subject = msg.subject
                msg_date = parser.parse(msg.date)
                msg_body = msg.body
                emails.append({
                    'filename': filename,
                    'from': msg_sender,
                    'to': msg_to,
                    'subject': msg_subject,
                    'date': msg_date,
                    'body': msg_body
                })
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    emails.sort(key=lambda x: x['date'])
    combined_text = ""
    for email in emails:
        combined_text += (
            f"From: {email['from']}\n"
            f"To: {email['to']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['date'].strftime('%d %b %Y, %I:%M %p')}\n"
            f"Body: {email['body']}\n"
            "-----\n"
        )
    summary = ollama_llm("Summarize this email conversation in a point-wise manner.", combined_text)
    return summary

def multimodal_chatbot(message, history, files=None):
    
    response = ""
    #file = files[0] if files else None  # Only process the first file for simplicity
    file = files
    print(files)

    if file is not None:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            text_splitter, vectorstore, retriever = process_pdf(file.name)
            if text_splitter is None:
                response = "Failed to process PDF."
            else:
                if message.strip():
                    response = rag_chain(message, text_splitter, vectorstore, retriever)
                else:
                    docs = retriever.get_relevant_documents("")
                    combined_text = combine_docs(docs)
                    response = ollama_llm("Summarize this document", combined_text)
        elif ext in [".xls", ".xlsx"]:
            response = process_excel(file.name)
        elif ext in [".png", ".jpg", ".jpeg"]:
            response = process_image(file.name)
        elif ext == ".msg":
            temp_dir = "./temp_emails"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            response = read_msg_emails_and_summarize(temp_dir)
        else:
            response = "Unsupported file type. Please upload PDF, Excel, image, or .msg files."
    else:
        response = ollama_llm(message, "")

    return response

chatbot_ui = gr.ChatInterface(
    multimodal_chatbot,
    chatbot=gr.Chatbot(
        label="Multimodal Chatbot",
        avatar_images=("C:\Users\lenovo\Documents\vs-code\github\AI-Agents-Chatbot\AI-Agents-Chatbot\images\human.jpg", "C:\Users\lenovo\Documents\vs-code\github\AI-Agents-Chatbot\AI-Agents-Chatbot\images\robot.png")
    ),
    textbox=gr.Textbox(
        placeholder="Type your question or upload a file...",
        container=False,
    ),
    additional_inputs=[
        gr.File(
            file_types=[".pdf", ".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".msg"],
            label="Upload PDF, Excel, Image, or .msg Email",
            show_label=True,
            interactive=True,
            visible=True,
#            multiple=True,
        )
    ],
    title="Multi-Modal Chatbot with Email Summarization",
    description="Ask questions or upload files (PDF, Excel, Images, Outlook .msg) for analysis and summarization.",
    theme="soft",
)

if __name__ == "__main__":
    chatbot_ui.launch()
