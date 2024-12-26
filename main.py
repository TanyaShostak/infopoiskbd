from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from load_data import load_all_data  # Импортируем функцию загрузки данных
from models import Base, Quote  # Импортируем модель Quote

DATABASE_URL = "postgresql+asyncpg://username:password@localhost/dbname"  # Замените на ваши данные

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # Создание таблиц
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

# Загрузка данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/quotes/", response_model=List[Quote])
def read_quotes(db: Session = Depends(get_db)):
    return db.query(Quote).all()

@app.post("/quotes/", response_model=Quote)
def add_quote(quote: Quote, db: Session = Depends(get_db)):
    db.add(quote)
    db.commit()
    db.refresh(quote)
    return quote

def update_indices(db: Session):
    global tfidf_matrix, bert_embeddings
    # Получаем все цитаты из базы данных
    quotes = db.query(Quote).all()
    
    # Обновляем TF-IDF
    tfidf_matrix = vectorizer.fit_transform([quote.content for quote in quotes])
    
    # Обновляем BERT
    bert_embeddings = []
    for quote in quotes:
        inputs = tokenizer(quote.content, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        bert_embeddings.append(embedding)
    bert_embeddings = np.array(bert_embeddings)

@app.get("/search/tfidf/")
def search_tfidf(query: str, db: Session = Depends(get_db)):
    start_time = time.time()
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # 5 наиболее похожих
    results = [{"quote": quotes[i], "relevance": similarities[i]} for i in top_indices]
    elapsed_time = time.time() - start_time
    return {"results": results, "time": elapsed_time}

@app.get("/search/bert/")
def search_bert(query: str, db: Session = Depends(get_db)):
    start_time = time.time()
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    similarities = cosine_similarity([query_embedding], bert_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # 5 наиболее похожих
    results = [{"quote": quotes[i], "relevance": similarities[i]} for i in top_indices]
    elapsed_time = time.time() - start_time
    return {"results": results, "time": elapsed_time}

@app.get("/search/tags/")
def search_by_tags(tag: str, db: Session = Depends(get_db)):
    start_time = time.time()
    results = db.query(Quote).filter(Quote.tags.contains(tag)).all()  # Поиск по тегам
    elapsed_time = time.time() - start_time
    return {"results": results, "time": elapsed_time}

@app.post("/recommendations/")
def recommend_quotes(quote_ids: List[str], db: Session = Depends(get_db)):
    if not quote_ids:
        raise HTTPException(status_code=400, detail="No quote IDs provided.")

    # Получаем эмбеддинги для выбранных цитат
    selected_embeddings = []
    for quote_id in quote_ids:
        matching_quote = db.query(Quote).filter(Quote.id == quote_id).first()
        if not matching_quote:
            raise HTTPException(status_code=404, detail=f"Quote with ID {quote_id} not found.")
        
        inputs = tokenizer(matching_quote.content, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        selected_embeddings.append(embedding)

    selected_embeddings = np.array(selected_embeddings)

    # Вычисляем средний вектор для выбранных цитат
    mean_embedding = selected_embeddings.mean(axis=0).reshape(1, -1)

    # Вычисляем косинусное сходство с остальными цитатами
    similarities = cosine_similarity(mean_embedding, bert_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # 5 наиболее похожих
    results = [{"quote": quotes[i], "relevance": similarities[i]} for i in top_indices]

    return {"results": results}