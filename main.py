from bpReviewClassifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from bpReviewClassifier.pipeline.stage02_data_transformation import DataTransformationPipeline
from bpReviewClassifier.logging import logger

STAGE_NAME='Data Ingestion Stage'
try:
    logger.info(f'Starting {STAGE_NAME}')
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f'{STAGE_NAME} completed successfully')
except Exception as e:
    raise e
STAGE_NAME='Data Transformation Stage'
try:
    logger.info(f'Starting {STAGE_NAME}')
    data_transformation=DataTransformationPipeline()
    data_transformation.main()
    logger.info(f'{STAGE_NAME} completed successfully')
except Exception as e:
    raise e