from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    password: str

class CreateUserRequestDto(BaseModel):
    name: str
    email: str
    password: str

class UserResponseDto(BaseModel):
    id: int
    name: str
    email: str