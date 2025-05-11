import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import requests
import re
import os
import shutil
import pandas as pd
from PIL import Image
#import pytesseract
import json
import datetime

# MSAL for Outlook authentication
from msal import PublicClientApplication

# ----------- Configuration ------------

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")

#Insert a call to Mistral LLM

# Outlook MSAL config - fill these with your Azure AD app info
CLIENT_ID = "your-client-id"
TENANT_ID = "your-tenant-id"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Mail.Read"]

# ----------- Outlook Authentication -----------

app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)

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
'''
def process_image(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No image file found."

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in the image."
        return ollama_llm("Summarize this extracted text from image", text)
    except Exception as e:
        return f"Error processing image: {str(e)}"
'''
# ----------- Outlook Email Summarization -----------

def get_emails_by_subject(subject):
    try:
        access_token = get_access_token()
    except Exception as e:
        return f"Authentication error: {str(e)}"

    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "$search": f'"subject:{subject}"',
        "$top": 50
    }
    response = requests.get("https://graph.microsoft.com/v1.0/me/messages", headers=headers, params=params)
    if response.status_code != 200:
        return f"Error fetching emails: {response.status_code} - {response.text}"

    emails = response.json().get("value", [])
    if not emails:
        return "No emails found with that subject."

    combined_text = "\n\n---\n\n".join(email.get("body", {}).get("content", "") for email in emails)
    # Optionally strip HTML tags here if needed

    summary = ollama_llm(f"Summarize emails with subject: {subject}", combined_text)
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
        #elif ext in [".png", ".jpg", ".jpeg"]:
        #    response = process_image(file.name)
        else:
            response = "Unsupported file type. Please upload PDF, Excel, or image files."
    else:
        # No file uploaded, check if user wants email summarization
        if user_message.lower().startswith("summarise my email with subject"):
            # Extract subject from command
            subject = user_message[len("summarise my email with subject"):].strip()
            if not subject:
                response = "Please provide a subject after the command."
            else:
                response = get_emails_by_subject(subject)
        else:
            response = "Please upload a file or enter a command like 'Summarise my email with Subject XYZ'."

    # Update chat history with user and bot messages
    chat_history.append((user_message, response))
    return chat_history, None  # Clear file upload after processing

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
