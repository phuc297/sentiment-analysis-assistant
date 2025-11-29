import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
import config
from underthesea import word_tokenize
from yaspin import yaspin
import time

LABELS = ["negative", "positive", "neutral"]

model_path = os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model.eval()
    yield
    del model
    del tokenizer


def predict_sentiment(sentence: str):
    sentence = word_tokenize(sentence, format="text")
    inputs = tokenizer(sentence, padding=True,
                       truncation=True, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)

        probabilities = out.logits.softmax(dim=-1).squeeze().tolist()

        pred_index = np.argmax(probabilities)
        predicted_label = LABELS[pred_index]

        prob_dict = {
            "negative": round(probabilities[0] * 100, 2),
            "positive": round(probabilities[1] * 100, 2),
            "neutral": round(probabilities[2] * 100, 2)
        }

        return predicted_label, prob_dict


app = FastAPI(title="PhoBERT Sentiment Analysis API", lifespan=lifespan)


class InputText(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    probabilities: dict


@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment_endpoint(data: InputText):
    if not data.text:
        return {"label": "Error", "probabilities": {"message": "Input text cannot empty."}}

    try:
        predicted_label, probabilities = predict_sentiment(data.text)

        return {
            "label": predicted_label,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"label": "Error", "probabilities": {"error": str(e)}}


@app.get("/")
def health_check():
    return {"status": "OK", "message": "Sentiment Analysis API is working."}

