import gradio as gr
import extract_msg
from dateutil import parser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
import requests
import re
import os
import shutil
import pandas as pd
from PIL import Image
import pytesseract
import json
import datetime
import easyocr
import extract_msg
from dateutil import parser


# MSAL for Outlook authentication
#from msal import PublicClientApplication

# ----------- Configuration ------------

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")


#Insert a call to Mistral LLM
llm = Ollama(model="mistral")
#print(llm("The first man on the summit of Mount Everest, the highest peak on Earth, was ..."))

# Outlook MSAL config - fill these with your Azure AD app info
#CLIENT_ID = "your-client-id"
#TENANT_ID = "your-tenant-id"
#AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
#SCOPES = ["Mail.Read"]

# ----------- Outlook Authentication -----------

#app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)
'''
def get_access_token():
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    else:
        flow = app.initiate_device_flow(scopes=SCOPES)
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception("Could not acquire access token")
'''

# ----------- PDF Processing -----------

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

# ----------- LLM Summarization -----------

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    prompt = "You are an AI Agent which can summarise the output in a point wise manner.\n" + formatted_prompt

    # Call LLM Agents 
    response_content = llm(formatted_prompt)
    if not response_content:
        return "No response from LLM."
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

# ----------- Excel Processing -----------

def process_excel(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No Excel file found."

    try:
        df_dict = pd.read_excel(file_path, sheet_name=None)
        texts = []
        for sheet_name, df in df_dict.items():
            texts.append(f"Sheet: {sheet_name}")
            # Convert dataframe to string, limit length to avoid huge input
            sheet_text = df.astype(str).agg('\n'.join, axis=1).str.cat(sep='\n')
            texts.append(sheet_text[:2000])  # truncate large sheets
        combined_text = "\n\n".join(texts)
        # Summarize combined text
        return ollama_llm("Summarize this Excel data", combined_text)
    except Exception as e:
        return f"Error processing Excel: {str(e)}"

# ----------- Image (Chart) Processing -----------
def process_image(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No image file found."

    try:
        reader = easyocr.Reader(['en'])  # You can add other language codes as needed
        result = reader.readtext(file_path, detail=0)  # detail=0 returns only text
        text = "\n".join(result)
        if not text.strip():
            return "No text detected in the image."
        return ollama_llm("Summarize this extracted text from image", text)
    except Exception as e:
        return f"Error processing image: {str(e)}"

# ----------- Outlook Email Summarization -----------
def read_msg_emails_and_summarize(directory):
    """
    Reads all .msg files in the given directory, sorts them by date, and summarizes the conversation.
    """
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

    # Sort emails by date
    emails.sort(key=lambda x: x['date'])

    # Combine email bodies for summarization
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

    # Use your LLM summarizer
    summary = ollama_llm("Summarize this email conversation in a point-wise manner.", combined_text)
    return summary


# ----------- Main Chatbot Handler -----------
def chatbot(user_message, chat_history, file):
    """
    user_message: str - user text input or command
    chat_history: list of tuples (user_msg, bot_msg)
    file: uploaded file (PDF, Excel, Image)
    """
    # Append user message to chat history
    chat_history = chat_history or []

    # Determine input type and handle accordingly
    response = ""
    if file is not None:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            # Process PDF
            text_splitter, vectorstore, retriever = process_pdf(file.name)
            if text_splitter is None:
                response = "Failed to process PDF."
            else:
                # Use RAG chain to answer question or summarize whole PDF if no question
                if user_message.strip():
                    response = rag_chain(user_message, text_splitter, vectorstore, retriever)
                else:
                    # Summarize entire PDF content
                    docs = retriever.get_relevant_documents("")  # get all docs
                    combined_text = combine_docs(docs)
                    response = ollama_llm("Summarize this document", combined_text)
        elif ext in [".xls", ".xlsx"]:
            response = process_excel(file.name)
        elif ext in [".png", ".jpg", ".jpeg"]:
            response = process_image(file.name)
        elif ext == ".msg":
            # Save uploaded file to a temp directory
            temp_dir = "./temp_emails"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            response = read_msg_emails_and_summarize(temp_dir)
            # Optionally, clean up temp_dir after processing
            # Update chat history with user and bot messages
            chat_history.append((user_message, response))
            return chat_history, None  # Clear file upload after processing
        else:
            response = "Unsupported file type. Please upload PDF, Excel, or image files."


# ----------- Gradio UI -----------

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Modal Chatbot with Email Summarization")

    chatbot_state = gr.State([])  # store chat history

    chatbot_ui = gr.Chatbot(label="Chatbot")

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="Type a message or command here...",
            lines=1,
            max_lines=5,
        )
        file_input = gr.File(
            label="Upload PDF, Excel, or Image (JPEG/PNG)",
            file_types=[".pdf", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"],
            interactive=True,
        )
        send_btn = gr.Button("Send")

    def submit_message(user_message, chat_history, file):
        return chatbot(user_message, chat_history, file)

    send_btn.click(
        submit_message,
        inputs=[txt_input, chatbot_state, file_input],
        outputs=[chatbot_ui, chatbot_state],
        queue=True,
    )
    # Clear file input after send
    send_btn.click(lambda: None, inputs=None, outputs=file_input)

    demo.launch()
