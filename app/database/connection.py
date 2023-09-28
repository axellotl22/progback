import os
from sqlalchemy import create_engine


def get_database_url():
    """ Erstellt die Datenbank URL je nach Enviroment """
    dev = os.environ['DEV_MODE']

    print(dev)

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

    return f"mysql+pymysql://progback:{pswd}@{host}:{port}/{schema}?charset=utf8mb4"


def get_engine():
    print(get_database_url())
    return create_engine(get_database_url())