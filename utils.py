from datetime import timedelta, datetime
from passlib.context import CryptContext
from jose import jwt
import os
import json

from schemas.auth import UserInfo

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encode_jwt


def user_entity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"],
        "is_verified": user["is_verified"],
        "password": user["password"],
        "phone": user["phone"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"]
    }


def get_email_from_token(token) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    email = payload.get("sub")
    return email


ALLOWED_EXTENSIONS = {'pdf', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def settings_check(num: str):
    if not os.path.exists(f"./settings/{num}"):
        os.makedirs(f"./settings/{num}")
        with open(f"./settings/{num}/settings.json", "w") as f:
            data = {"title": f"Bot {num}", "header": "",
                    "bot": "", "user": "", "model": "gpt-3.5-turbo"}
            json.dump(data, f)
