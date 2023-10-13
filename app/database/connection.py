import os

from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import pytest


@pytest.mark.skip(reason="Dont go on Prod Database")
def get_database_url():
    """ Erstellt die Datenbank URL je nach Enviroment """
    test_mode = os.environ.get('TEST_MODE')

    if test_mode == 'True' or test_mode is None:
        return False

    dev = os.environ['DEV_MODE']

    if dev == 'True':
        host = os.environ['DB_HOST']
        port = os.environ['DB_PORT']
        pswd = os.environ['DB_PW']
        schema = os.environ['DB_SCHEMA']
    else:
        host = os.environ['D_DB_HOST']
        port = os.environ['D_DB_PORT']
        pswd = os.environ['D_DB_PW']
        schema = os.environ['D_DB_SCHEMA']

    if host == '':
        return False

    return f"mysql+pymysql://progback:{pswd}@{host}:{port}/{schema}?charset=utf8mb4"


@pytest.mark.skip(reason="Dont go on Prod Database")
def get_async_database_url():
    """ Erstellt die async Datenbank URL je nach Enviroment """
    test_mode = os.environ.get('TEST_MODE')

    if test_mode == 'True' or test_mode is None:
        return False

    dev = os.environ['DEV_MODE']

    if dev == 'True':
        host = os.environ['DB_HOST']
        port = os.environ['DB_PORT']
        pswd = os.environ['DB_PW']
        schema = os.environ['DB_SCHEMA']
    else:
        host = os.environ['D_DB_HOST']
        port = os.environ['D_DB_PORT']
        pswd = os.environ['D_DB_PW']
        schema = os.environ['D_DB_SCHEMA']

    if host == '':
        return False

    return f"mysql+aiomysql://progback:{pswd}@{host}:{port}/{schema}?charset=utf8mb4"


@pytest.mark.skip(reason="Dont go on Prod Database")
def get_engine():
    test_mode = os.environ.get('TEST_MODE')

    if test_mode == "True" or test_mode is None:
        # Für Unit Tests
        if os.path.exists("test/test.db"):
            os.remove("test/test.db")
        return create_engine("sqlite+aiosqlite:///test/test.db", connect_args={"check_same_thread": False})

    db_url = get_database_url()
    return create_engine(db_url)


@pytest.mark.skip(reason="Dont go on Prod Database")
def get_async_engine():
    test_mode = os.environ.get('TEST_MODE')

    if test_mode == "True" or test_mode is None:
        # Für Unit Tests
        if os.path.exists("test/test.db"):
            os.remove("test/test.db")
        return create_async_engine("sqlite+aiosqlite:///test/test.db", connect_args={"check_same_thread": False})

    db_url = get_async_database_url()

    return create_async_engine(db_url)


engine = get_engine()
async_engine = get_async_engine()


@pytest.mark.skip(reason="Dont go on Prod Database")
def get_db():
    """
    Gibt eine Instanz der Datenbankverbindung zurück
    """
    database = Session(engine)

    try:
        yield database
    finally:
        database.close()


@pytest.mark.skip(reason="Dont go on Prod Database")
async def get_async_db():
    """
    Gibt eine Asynchrone Instanz der Datenbank zurück
    """

    database = AsyncSession(async_engine)

    try:
        yield database
    finally:
        await database.close()
