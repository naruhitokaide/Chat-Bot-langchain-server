from fastapi import APIRouter, HTTPException, Depends, status, Request, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_jwt_auth import AuthJWT
from typing import Annotated
from pydantic import EmailStr
from datetime import datetime
from random import randbytes
import hashlib
from jose import JWTError
from google.oauth2 import id_token
from google.auth.transport import requests

from core.database import db
from utils import get_password_hash, verify_password, create_access_token, user_entity, get_email_from_token
from core.email import Email
from schemas.auth import UserRegisterRequest, UserInfo, UserGoogle

router = APIRouter(prefix="/api/auth", tags=["auth"])

users_collection = db["users"]


@router.post("/register", description="register user")
async def register_user(user: UserRegisterRequest, request: Request):
    # Check if the user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Account already exist")

    # Hash the password and insert the new user into the database
    password_hash = get_password_hash(user.password)

    user_dict = user.dict()
    user_dict["password"] = password_hash
    user_dict["role"] = "user"
    user_dict["is_verified"] = False
    user_dict["created_at"] = datetime.utcnow()
    user_dict["updated_at"] = user_dict["created_at"]

    result = users_collection.insert_one(user_dict)
    new_user = users_collection.find_one({"_id": result.inserted_id})

    try:
        token = randbytes(10)
        hashed_code = hashlib.sha256()
        hashed_code.update(token)
        verification_code = hashed_code.hexdigest()
        users_collection.find_one_and_update({"_id": result.inserted_id}, {"$set": {
            "verification_code": verification_code, "updated_at": datetime.utcnow()}})
        print(token.hex())
        url = f"https://coral-app-3bimw.ondigitalocean.app/verify/{token.hex()}"
        await Email(user_entity(new_user), url, [EmailStr(user.email)]).sendVerificationCode()
    except Exception as error:
        users_collection.find_one_and_update({"_id": result.inserted_id}, {
                                             "$set": {"verification_code": None, "updated_at": datetime.utcnow()}})
        print(error)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="There was error sending email")
    return {"status": "success", "message": "Verification token successfully sent"}


@router.get('/verifyemail/{token}')
async def verify_me(token: str):
    hashed_code = hashlib.sha256()
    hashed_code.update(bytes.fromhex(token))
    verification_code = hashed_code.hexdigest()
    result = users_collection.find_one_and_update({"verification_code": verification_code}, {"$set": {
                                                  "verification_code": None, "is_verified": True, "updated_at": datetime.utcnow()}}, new=True)
    if not result:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid verification code or account already verified")
    return {
        "status": "success",
        "message": "Account verified successfully"
    }


@router.post("/login", description="login user")
async def login_user(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = users_collection.find_one({"email": form_data.username})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username", headers={"WWW-Authenticate": "Bearer"})
    if not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect password", headers={"WWW-Authenticate": "Bearer"})
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "user": UserInfo(**user)}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        email = get_email_from_token(token)
        if email is None:
            raise credentials_exception
        user = users_collection.find_one({"email": email})
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception


@router.get("/user", response_model=UserInfo)
async def read_user(current_user: Annotated[UserInfo, Depends(get_current_user)]):
    return current_user


GOOGLE_CLIENT_ID = "504867464977-bnrvev217iqfpb02mg8nb7v1te3e73pu.apps.googleusercontent.com"


@router.post('/login/google')
async def login_google(token: Annotated[str, Body(embed=True)]):
    try:
        # Verify the access token with Google
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID)

        user = users_collection.find_one({"email": idinfo["email"]})
        if not user:
            present = datetime.utcnow()
            users_collection.insert_one({
                "email": idinfo["email"],
                "full_name": idinfo["name"],
                "is_verified": True,
                "role": "user",
                "created_at": present,
                "updated_at": present
            })
        user = users_collection.find_one({"email": idinfo["email"]})
        access_token = create_access_token(data={"sub": user["email"]})
        return {"access_token": access_token, "user": UserInfo(**user)}
    except ValueError:
        #  Handle invalid tokens
        return {"error": "Invalid token"}
