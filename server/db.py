import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app.db'))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    preferences = relationship('Preference', back_populates='user', uselist=False)
    runs = relationship('Run', back_populates='user')


class Preference(Base):
    __tablename__ = 'preferences'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    params_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='preferences')


class Run(Base):
    __tablename__ = 'runs'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    run_uid = Column(String(64), index=True, nullable=False)
    dataset = Column(String(50), nullable=False)
    params_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship('User', back_populates='runs')


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
