from pymongo import MongoClient
import os
from dotenv import load_dotenv
from core.config import db_name

load_dotenv()

client = MongoClient(os.getenv("MONGO_SERVER_URI"))
db = client.get_database(db_name)
