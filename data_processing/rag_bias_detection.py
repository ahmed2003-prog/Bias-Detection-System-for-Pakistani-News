import chromadb
import pandas as pd
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
import enchant

class BiasDetectionSystem:
    def __init__(self, db_path="./chroma_db", emb_model="sentence-transformers/all-MiniLM-L6-v2",
                 roberta_model="cardiffnlp/twitter-roberta-base-sentiment"):
        self.emb_model = SentenceTransformer(emb_model)
        self.dictionary = enchant.Dict("en_US")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="my_collection")
        self.whitelist = {
            # Political Parties
            "pmln", "pti", "ppp", "mqm", "jui-f", "ji", "anp", "bnp", "tlyp", "aml", "juif", "tlp", "psp", "gda", "pta",  
            "pak sarzameen party", "grand democratic alliance", "pakistan muslim league", "pakistan tehreek-e-insaf",  
            "pakistan people's party", "mutahida qaumi movement", "pakistan democratic movement", "balochistan national party",  
            "national party", "pakistan awami tehreek", "pakistan rah-e-haq party", "mqm-p", "mqm-l", "sunni tehreek",  

            # Government & Judiciary
            "sc", "supreme court", "hc", "high court", "ecp", "election commission", "parliament", "senate", "assembly",  
            "speaker", "deputy speaker", "law ministry", "attorney general", "chief justice", "district court",  

            # Political Figures  
            "nawaz", "imran", "bhutto", "benazir", "zardari", "bilawal", "shahbaz", "maryam", "asad umar", "fawad chaudhry",  
            "shah mahmood qureshi", "chaudhry pervaiz elahi", "rana sanaullah", "murad saeed", "pervaiz khattak",  
            "jahangir tareen", "sheikh rashid", "fazlur rehman", "hafiz saeed", "asad qaiser",  

            # Law Enforcement & Security  
            "fbr", "nab", "fia", "isi", "army", "coas", "dg isi", "dg ispr", "ispr", "pak army", "rangers",  
            "pak navy", "pak air force", "igp", "ssp", "dsp", "shc", "phc", "bfc", "fc", "ctd", "counter terrorism department",  

            # Government Titles & Positions  
            "cm", "pm", "mna", "mpa", "mps", "president", "governor", "cabinet", "foreign minister", "interior minister",  
            "defense minister", "chief minister", "opposition leader", "finance minister", "planning commission",  

            # Economic & Financial Institutions  
            "state bank", "sbp", "imf", "world bank", "finance division", "secp", "oecd", "budget", "tax", "inflation",  
            "trade deficit", "stock exchange", "psx", "kse", "forex", "remittances",  

            # Media & Regulatory Bodies  
            "pemra", "pta", "ppra", "ndma", "pmd", "met office", "information ministry", "journalist", "anchor",  
            "media ban", "press conference", "freedom of press",  

            # Other Relevant Terms  
            "cpec", "bri", "gawadar", "karot", "dams", "mangla", "tarbela", "diamir", "motorway", "highway", "tunnel",  
            "chinese investment", "foreign direct investment", "trade agreements", "free trade zone"
        }
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model)

    def is_gibberish(self, text):
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return True
        valid_word_count = sum(1 for word in words if self.dictionary.check(word) or word.lower() in self.whitelist or word.upper() in self.whitelist)
        return (valid_word_count / len(words)) < 0.15

    def query_bias(self, query: str, n_results: int = 5):
        query_embedding = self.emb_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results

    def preprocess_dataset(self, filepath="labeled_news_data.csv"):
        df = pd.read_csv(filepath)
        df["full_text"] = df["Story Heading"].fillna("") + ". " + df["Story Excerpt"].fillna("")
        df = df[df["full_text"].str.strip() != ""]
        df["embedding"] = df["full_text"].apply(lambda x: self.emb_model.encode(x).tolist())
        return df

    def upload_to_chromadb(self, df, batch_size=4000):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            self.collection.upsert(
                ids=[str(j) for j in batch.index],
                embeddings=batch["embedding"].tolist(),
                documents=batch["full_text"].tolist(),
                metadatas=batch[["Section", "Date", "Source", "Sentiment", "Bias_Score", "Bias_Label"]].to_dict(orient="records"),
            )
        print("✅ Dataset successfully uploaded to ChromaDB!")

    def predict_bias(self, text):
        if not text.strip() or self.is_gibberish(text):
            return {"bias_label": "Unknown", "bias_score": 0, "bias_classification": "Out of Scope", "confidence": 0.0}

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=-1).squeeze().tolist()
        labels = ["Negative", "Neutral", "Positive"]
        max_index = probabilities.index(max(probabilities))
        bias_score = round(probabilities[max_index] * 100)

        bias_classification = (
            "Highly Unbiased" if bias_score <= 10 else
            "Slightly Unbiased" if bias_score <= 20 else
            "Moderately Unbiased" if bias_score <= 40 else
            "Neutral" if bias_score <= 50 else
            "Moderately Biased" if bias_score <= 60 else
            "Biased" if bias_score <= 70 else
            "Highly Biased"
        )

        return {
            "bias_label": labels[max_index],
            "bias_score": bias_score,
            "bias_classification": bias_classification,
            "confidence": probabilities[max_index]
        }
