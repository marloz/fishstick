from datetime import datetime, timedelta

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from src.data import ticker_pipe
from src.pipeline.calculate_features import WINDOW_LENGTHS, calculate_features

# run using uvicorn api.app:app --reload
app = FastAPI()


@app.get("/")
def index():
    return JSONResponse("Hi from Fishstick!")


@app.post("/predict")
def predict(response: Response, ticker: str) -> JSONResponse:
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        end_date = end_date.strftime("%F")
        start_date = start_date.strftime("%F")
        return (
            ticker_pipe([ticker], start_date, end_date)
            .pipe(calculate_features, window_lengths=WINDOW_LENGTHS)
            .loc[lambda x: x["Date"] == x["Date"].max()]
            .to_json()
        )

    except Exception as e:
        response.status_code = 500
        return JSONResponse(content={"err": str(e)})
