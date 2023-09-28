from sqlalchemy import Column, String, create_engine, BigInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.database.connection import get_engine

# Datenbank-Konfiguration
Base = declarative_base()
engine = get_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "user"
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    username = Column(String(80), unique=True, index=True)
    email = Column(String(128), unique=True, index=True)


# Tabelle erstellen
Base.metadata.create_all(bind=engine)
session = Session(bind=engine)


def create_user():
    new_user = User(username='mh', email='moritz.holtz@hs-osnabrueck.de')
    session.add(new_user)
    session.commit()
