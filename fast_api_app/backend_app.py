from data_processing.rag_bias_detection import BiasDetectionSystem
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import datetime
import uvicorn


bias_system = BiasDetectionSystem()


app = FastAPI()

# Allow CORS for all origins (useful for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
        results = bias_system.collection.get(limit=1000, include=["metadatas"])
        bias_scores = [meta["Bias_Score"] for meta in results["metadatas"] if meta and "Bias_Score" in meta]
        dates = [meta["Date"] for meta in results["metadatas"] if meta and "Date" in meta]

        trend_data = {}
        for date, score in zip(dates, bias_scores):
            date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            trend_data.setdefault(date, []).append(score)

        bias_trends_over_time = {date.isoformat(): sum(scores) / len(scores) for date, scores in trend_data.items()}

        return {"bias_trends": bias_trends_over_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/article_details")
def article_details(article_id: str):
    try:
        results = bias_system.collection.get(ids=[article_id], include=["documents", "metadatas"])
        if not results["documents"]:
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
