from sqlalchemy import Column, String, create_engine, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.database.connection import get_database_url

# Datenbank-Konfiguration
Base = declarative_base()
db_url = get_database_url()
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Benutzermodell
class User(Base):
    __tablename__ = "user"
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    username = Column(String(80), unique=True, index=True)
    email = Column(String(128), unique=True, index=True)


# Tabelle erstellen
Base.metadata.create_all(bind=engine)
session = Session(bind=engine)


def create_user():
    print(db_url)
    new_user = User(username='ar', email='alex.richter39@gmail.com')
    session.add(new_user)
    session.commit()
