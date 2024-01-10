import json
import os

import requests
import streamlit as st

if __name__ == "__main__":
    # Define how to reach model endpoints
    host: str = os.getenv("MODEL_HOST", "127.0.0.1")
    port: str = os.getenv("PORT", "8000")
    base_url: str = f"http://{host}:{port}"

    try:
        ticker: str = st.text_input(label="Input ticker", value="", max_chars=5)
        clicked_predict: bool = st.button(label="Predict")

        if clicked_predict:
            validate_request_url = base_url + f"/validate?ticker={ticker}"
            validate_resp: dict = json.loads(requests.post(validate_request_url).content)
            if validate_resp["ticker"] == "Not found!":
                st.write(f"Ticker {ticker} not found!")
            else:
                features_request_url = base_url + f"/features?ticker={ticker}"
                features_resp: str = json.loads(requests.post(features_request_url).content)
                st.write(features_resp)

                predict_request_url = base_url + f"/predict?features={features_resp}"
                predict_resp: str = json.loads(requests.post(predict_request_url).content)
                st.write(predict_resp)

    except Exception as e:
        st.error(e)
