"""User Router"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.connection import get_db
from app.entitys.user import User


router = APIRouter()

class UserResponse(BaseModel):
    """Response Model For Users"""
    id: int
    username: str
    email: str

@router.get("/", response_model=List[UserResponse])
async def get_users(database: Session = Depends(get_db)):
    """Gets all Users"""
    users =  database.query(User).all()

    clean_users = []

    for user in users:
        clean_users.append(UserResponse(id = user.id, username=user.username, email= user.email))

    return clean_users
