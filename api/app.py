import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import yfinance
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from src.data import ticker_pipe
from src.model import load_model
from src.pipeline.calculate_features import WINDOW_LENGTHS, calculate_features

MODEL_PATH = "models/model.dill"
LOOKBACK_WINDOW = 365

# run using uvicorn api.app:app --reload
app = FastAPI()


@app.get("/")
def index():
    return JSONResponse("Hi from Fishstick!")


@app.post("/features")
def get_current_features_for_ticker(ticker: str) -> JSONResponse:
    try:
        logger.info(f"Getting features for ticker: {ticker}")
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=LOOKBACK_WINDOW)).strftime("%F")
        return (
            ticker_pipe([ticker.upper()], start_date, end_date.strftime("%F"))
            .pipe(calculate_features, window_lengths=WINDOW_LENGTHS)
            .loc[lambda x: x["Date"] == x["Date"].max()]
            .to_json(orient="records")
        )
    except Exception as e:
        return JSONResponse(content={"err": str(e)})


@dataclass
class Result:
    probability: float


@app.post("/predict")
def predict(features: str) -> JSONResponse:
    try:
        logger.info(f"Loading model {MODEL_PATH}")
        model = load_model(MODEL_PATH)

        feature_dict = json.loads(features)
        features_df = pd.DataFrame.from_dict(feature_dict)

        pred = float(model.predict_proba(features_df)[0:, 1])
        res = Result(probability=pred)
        logger.info(f"Predicted probability: {res}")
        return JSONResponse(content=json.dumps(res.__dict__))

    except Exception as e:
        return JSONResponse(content={"err": str(e)})


@app.post("/validate")
def validate_ticker(ticker: str) -> JSONResponse:
    try:
        if len(yfinance.Ticker(ticker).history_metadata) == 0:
            return JSONResponse(content={"ticker": "Not found!"})
        return JSONResponse(content={"ticker": ticker})

    except Exception as e:
        return JSONResponse(content={"err": str(e)})
