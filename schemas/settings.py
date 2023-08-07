from pydantic import BaseModel


class BasicSettings(BaseModel):
    title: str
    model: str
