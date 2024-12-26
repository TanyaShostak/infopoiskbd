from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

class Author(Base):
    __tablename__ = 'authors'
    id = Column(String, primary_key=True)
    name = Column(String)
    bio = Column(String)
    website = Column(String)

class Quote(Base):
    __tablename__ = 'quotes'
    id = Column(String, primary_key=True)
    content = Column(String)
    author_id = Column(String, ForeignKey('authors.id'))
    author = relationship("Author")
    vector = Column(ARRAY(Float))  # Векторное представление цитаты

class Tag(Base):
    __tablename__ = 'tags'
    id = Column(String, primary_key=True)
    name = Column(String)

class Tagging(Base):
    __tablename__ = 'tagging'
    quote_id = Column(String, ForeignKey('quotes.id'), primary_key=True)
    tag_id = Column(String, ForeignKey('tags.id'), primary_key=True)

# Создание базы данных
engine = create_engine('postgresql://postgres:password@localhost/postgres')
Base.metadata.create_all(engine)