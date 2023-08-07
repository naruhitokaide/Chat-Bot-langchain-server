import json
import os
from fastapi import FastAPI, WebSocket, UploadFile, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from pydantic import BaseModel

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader, TextLoader
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = './files'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = FastAPI()
app.mount("/settings", StaticFiles(directory="settings"), name="static")
app.mount("/store", StaticFiles(directory="store"), name="static")

origins = [
    "https://seahorse-app-kbdql.ondigitalocean.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post('/api/upload/{num}')
async def upload(file: UploadFile, num: str):
    path = Path(UPLOAD_FOLDER) / file.filename

    if file and allowed_file(file.filename):
        path.write_bytes(await file.read())
        fileext = file.filename.rsplit('.', 1)[1].lower()
        if (fileext == 'pdf'):
            reader = PdfReader(path)
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len)
            texts = text_splitter.split_text(raw_text)
            embeddings = OpenAIEmbeddings()
            if os.path.exists(f"./store/{num}/index.faiss"):
                docsearch = FAISS.load_local(f"./store/{num}", embeddings)
                docsearch.add_texts(texts)
            else:
                docsearch = FAISS.from_texts(texts, embeddings)
            docsearch.save_local(f"./store/{num}")
        else:
            loader = TextLoader(path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len)
            split_docs = loader.load_and_split(text_splitter)
            embeddings = OpenAIEmbeddings()
            if os.path.exists(f"./store/{num}/index.faiss"):
                docsearch = FAISS.load_local(f"./store/{num}", embeddings)
                docsearch.add_documents(split_docs)
            else:
                docsearch = FAISS.from_documents(split_docs, embeddings)
            docsearch.save_local(f"./store/{num}")

        return {"state": "success"}
    return {"state": "error", "message": "Invalid file format"}


@app.post('/api/youtube/train/{num}')
async def train_youtube(num: int, url: str = Body(embed=True)):
    loader = YoutubeLoader.from_youtube_channel(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    if (os.path.exists(f"./store/{num}/index.faiss")):
        docsearch = FAISS.load_local(f"./store/{num}", embeddings)
        docsearch.add_documents(texts)
    else:
        docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(f"./store/{num}")
    return {"state": "success"}

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)


@app.websocket("/api/chat/{num}")
async def chat(websocket: WebSocket, num: str):
    await websocket.accept()
    llm = OpenAI(temperature=0)
    settings_check(num)
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    if data["model"] != "text-davinci-003":
        llm = ChatOpenAI(model_name=data["model"], temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="human_input")
    chain = load_qa_chain(llm=llm, chain_type="stuff",
                          memory=memory, verbose=True, prompt=prompt)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local(f"./store/{num}", embeddings)

    while True:
        try:
            query = await websocket.receive_text()
            docs = docsearch.similarity_search(query)
            completion = chain.run(input_documents=docs, human_input=query)
            await websocket.send_text(completion)
        except WebSocketDisconnect:
            break


def settings_check(num: str):
    if not os.path.exists(f"./settings/{num}"):
        os.makedirs(f"./settings/{num}")
        with open(f"./settings/{num}/settings.json", "w") as f:
            data = {"title": f"Bot {num}", "header": "",
                    "bot": "", "user": "", "model": "gpt-3.5-turbo"}
            json.dump(data, f)


class Item(BaseModel):
    title: str
    model: str


@app.post("/api/header-change/{num}")
async def header_change(num: str, item: Item):
    item_dict = item.dict()
    print(item_dict)
    settings_check(num)
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["title"] = item_dict["title"]
    data["model"] = item_dict["model"]
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)
    return {"status": "success"}


@app.post("/api/header-upload/{num}")
async def header_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'header.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["header"] = f"header.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@app.post("/api/botimg-upload/{num}")
async def botimg_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'bot.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["bot"] = f"bot.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@app.post("/api/userimg-upload/{num}")
async def userimg_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'user.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["user"] = f"user.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@app.get("/api/settings/{num}")
async def get_settings(num: str):
    settings_check(num)
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
        return data

if __name__ == "__main__":
    if not os.path.exists(f"./{UPLOAD_FOLDER}"):
        os.makedirs(f"./{UPLOAD_FOLDER}")
    if not os.path.exists("./settings"):
        os.makedirs("./settings")
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
    
