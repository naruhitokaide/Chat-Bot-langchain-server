from pydantic import BaseModel


class UserRegisterRequest(BaseModel):
    email: str
    full_name: str
    phone: str
    password: str


class UserInfo(BaseModel):
    email: str
    full_name: str
    phone: str | None
    role: str


class UserGoogle(BaseModel):
    sub: str
    name: str
    email: str
    picture: str
