from data_processing.rag_bias_detection import BiasDetectionSystem
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
from collections import defaultdict

bias_system = BiasDetectionSystem()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    news_text: str

class DatasetUpload(BaseModel):
    file_path: str

@app.post("/analyze_news_bias")
def analyze_news_bias(news: NewsInput):
    if not news.news_text.strip():
        raise HTTPException(status_code=400, detail="News text cannot be empty.")

    if bias_system.is_gibberish(news.news_text):
        return {
            "news_text": news.news_text,
            "bias_analysis": {
                "bias_label": "Unknown",
                "bias_score": 0,
                "bias_classification": "Out of Scope",
                "confidence": 0.0
            },
            "message": "The input was classified as gibberish or not relevant."
        }

    bias_prediction = bias_system.predict_bias(news.news_text)

    return {
        "news_text": news.news_text,
        "bias_analysis": bias_prediction
    }

@app.get("/query_bias")
def query_bias(query: str, n_results: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if bias_system.is_gibberish(query):
        return {"query": query, "message": "Query classified as gibberish or irrelevant."}

    results = bias_system.query_bias(query, n_results)
    bias_prediction = bias_system.predict_bias(query)

    related_articles = []
    if results.get("documents") and results["documents"][0]:
        related_articles = [{"document": result, "metadata": metadata}
                            for result, metadata in zip(results["documents"][0], results["metadatas"][0])]

    return {
        "query": query,
        "bias_analysis": bias_prediction,
        "related_articles": related_articles if related_articles else "No relevant articles found."
    }

@app.post("/upload_dataset")
def upload_dataset(upload: DatasetUpload):
    try:
        df = bias_system.preprocess_dataset(upload.file_path)
        bias_system.upload_to_chromadb(df)
        return {"message": "Dataset uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bias_trends")
def bias_trends():
    try:
        if not hasattr(bias_system, "collection") or bias_system.collection is None:
            raise HTTPException(status_code=500, detail="Bias detection system is not initialized.")

        results = bias_system.collection.get(limit=1000, include=["metadatas"])

        if not results.get("metadatas"):
            raise HTTPException(status_code=500, detail="No metadata found in the database.")

        bias_scores = []
        dates = []

        for meta in results["metadatas"]:
            if meta and "Bias_Score" in meta and "Date" in meta:
                try:
                    date = datetime.datetime.strptime(meta["Date"], "%Y-%m-%d").date()
                    bias_score = float(meta["Bias_Score"])
                    bias_scores.append(bias_score)
                    dates.append(date)
                except (ValueError, TypeError):
                    continue

        trend_data = defaultdict(list)
        for date, score in zip(dates, bias_scores):
            trend_data[date].append(score)

        bias_trends_over_time = {
            date.isoformat(): sum(scores) / len(scores) for date, scores in sorted(trend_data.items())
        }

        if not bias_trends_over_time:
            return {"bias_trends": {}, "summary": "No bias data available."}

        min_bias = min(bias_trends_over_time.values())
        max_bias = max(bias_trends_over_time.values())
        trend_direction = "Increasing" if list(bias_trends_over_time.values())[-1] > list(bias_trends_over_time.values())[0] else "Decreasing"

        return {
            "bias_trends": bias_trends_over_time,
            "summary": {
                "min_bias": min_bias,
                "max_bias": max_bias,
                "trend_direction": trend_direction,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/article_details")
def article_details(article_id: str):
    try:
        if not hasattr(bias_system, "collection") or bias_system.collection is None:
            raise HTTPException(status_code=500, detail="Bias detection system is not initialized.")

        results = bias_system.collection.get(ids=[article_id], include=["documents", "metadatas"])

        if not results.get("documents") or not results["documents"][0]:
            raise HTTPException(status_code=404, detail="Article not found.")

        return {
            "article_id": article_id,
            "document": results["documents"][0],
            "metadata": results["metadatas"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
