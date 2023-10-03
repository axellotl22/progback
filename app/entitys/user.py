from fastapi import Depends
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy

from app.database.user_db import get_user_db

class UserRead(BaseUser):
    username: str
    pass


class UserCreate(BaseUserCreate):
    username: str
    pass


class UserUpdate(BaseUserUpdate):
    username: str
    pass


APP_SECRET = 'test'
VERIFICATION_SECRET = 'test2'

class UserManager(UUIDIDMixin, BaseUserManager):
    reset_password_token_secret = APP_SECRET
    verification_token_secret = VERIFICATION_SECRET


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


cookie_transport = CookieTransport(cookie_httponly=True, cookie_secure=True)


def get_jwt_strategy():
    return JWTStrategy(secret=APP_SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(name="jwt", transport=cookie_transport, get_strategy=get_jwt_strategy)


fastapi_users = FastAPIUsers(get_user_manager, [auth_backend])
active_user = fastapi_users.current_user(active=True)
