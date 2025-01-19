from bpReviewClassifier.config.configuartion import ConfigurationManager
from bpReviewClassifier.components.data_ingestion import DataIngestion
from bpReviewClassifier.logging import logger

class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()