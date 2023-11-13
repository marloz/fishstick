import hydra
from loguru import logger
from omegaconf import DictConfig

from src.config import GetDataConfig
from src.data import scrape_tickers, ticker_pipe
from src.utils import parse_dict_config


@hydra.main(config_path="../../config", config_name="get_data", version_base=None)
def main(config_: DictConfig) -> None:
    """Scrapes current stock tickers from wiki,
    then gets their price data from yahoo finance and stores in a
    parquet file."""
    config: GetDataConfig = parse_dict_config(GetDataConfig, config_)
    logger.info(f"Starting get data step, using config: \n{config}")
    tickers = scrape_tickers(
        config.ticker_config.url,
        config.ticker_config.ticker_limit,
    )
    df = ticker_pipe(
        tickers,
        config.yahoo_config.start_date,
        config.yahoo_config.end_date,
        config.yahoo_config.interval,
    )
    logger.info("Writing results")
    df.to_parquet(config.output_path, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
