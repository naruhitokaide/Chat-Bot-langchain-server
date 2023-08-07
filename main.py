from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routes import auth, settings, chat

load_dotenv()

app = FastAPI()

app.mount("/settings", StaticFiles(directory="settings"), name="static")
app.mount("/store", StaticFiles(directory="store"), name="static")

origins = ["http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(settings.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, port=9000)
    if not os.path.exists(f"./files"):
        os.makedirs(f"./files")
    if not os.path.exists("./settings"):
        os.makedirs("./settings")
