from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json

from utils import settings_check

router = APIRouter(prefix="/api", tags=["chat"])


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


@router.websocket("/chat/{num}")
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
