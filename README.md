# fishstick
E2E stock market prediction using machine learning

# Purpose
This project is intended for personal use/learning as a demo of E2E machine learning project with use case for stock market prediction.

# Outline

```mermaid
flowchart

    source[(Data source)]

    subgraph Development

        subgraph Feature Pipeline
            get_hist_data[get historical data] --> db[(Raw data)]
            db --> features[create features and target]
            features --> dataset[(Prepared data)]
        end

        subgraph Training Pipeline
            dataset --> train[train model]
            train --> registry[(Model registry)]
        end

    end

    source --> get_hist_data
    registry -- deploy --> Production

    subgraph Production

        subgraph API
            get_live_data[get live data] --> predict
        end

        subgraph UI
            expose_pred[expose prediction]
        end

        UI -- request prediction --> API
        API -- return prediction --> UI

        subgraph Monitoring
            data_quality[data quality]
            model_performance[model performance]
        end

        API -- send data to monitor --> Monitoring
    end

    source --> get_live_data
    Monitoring -- trigger retraining --> Development


```