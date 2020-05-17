from abc import ABC, abstractmethod
import logging.config

from utils.constants import LOG_CONFIG_PATH

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    _write_path = None
    _read_path = None

    @classmethod
    @abstractmethod
    def extract(cls):
        logger.info(f"Reading from {cls._read_path}")

    @classmethod
    @abstractmethod
    def transform(cls):
        logger.info("Transforming dataframe")

    @classmethod
    @abstractmethod
    def load(cls):
        logger.info(f"Writing to {cls._write_path}")

    @classmethod
    @abstractmethod
    def main(cls):
        cls.extract()
        cls.transform()
        cls.load()
