import os
from sqlalchemy import create_engine


def get_database_url():
    stabilty = os.environ['STABILITY']

    if stabilty == 'stable':
        host = os.environ['D_DB_HOST']
        port = os.environ['D_DB_PORT']
        pswd = os.environ['D_DB_PW']
        schema = os.environ['D_DB_SCHEMA']
    elif stabilty == 'dev':
        host = os.environ['DB_HOST']
        port = os.environ['DB_PORT']
        pswd = os.environ['DB_PW']
        schema = os.environ['DB_SCHEMA']
    else:
        raise Exception('Invalid Stability')

    return f"mysql+pymysql://progback:{pswd}@{host}:{port}/{schema}?charset=utf8mb4"
