import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Author, Quote, Tag, Base
from sentence_transformers import SentenceTransformer

# Создание сессии
engine = create_engine('postgresql://postgres:password@localhost/postgres')
Session = sessionmaker(bind=engine)
session = Session()

# Инициализация модели для векторов
model = SentenceTransformer('all-MiniLM-L6-v2')

# Загрузка авторов
with open('data/authors.json') as f:
    authors_data = json.load(f)
    for author in authors_data:
        author_entry = Author(
            id=author['id'],
            name=author['name'],
            bio=author.get('bio', ''),
            website=author.get('website', '')
        )
        session.add(author_entry)

# Загрузка цитат
with open('data/quotes.json') as f:
    quotes_data = json.load(f)
    for quote in quotes_data:
        vector = model.encode(quote['content']).tolist() 
        quote_entry = Quote(
            id=quote['id'],
            content=quote['content'],
            author_id=quote['authorId'],
            vector=vector  # Сохранение вектора
        )
        session.add(quote_entry)

# Загрузка тегов
with open('data/tags.json') as f:
    tags_data = json.load(f)
    for tag in tags_data:
        tag_entry = Tag(
            id=tag['id'],
            name=tag['name']
        )
        session.add(tag_entry)

session.commit()
session.close()
