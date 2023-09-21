import os


def get_database_url():
    dev = os.environ.get('DEV_MODE')

    if dev:
        host = os.environ.get('DB_HOST')
        port = os.environ.get('DB_PORT')
        pswd = os.environ.get('DB_PW')
        schema = os.environ.get('DB_SCHEMA')
    else:
        host = os.environ.get('D_DB_HOST')
        port = os.environ.get('D_DB_PORT')
        pswd = os.environ.get('D_DB_PW')
        schema = os.environ.get('D_DB_SCHEMA')

    return f"mysql+pymysql://progback:{pswd}@{host}:{port}/{schema}?charset=utf8mb4"
