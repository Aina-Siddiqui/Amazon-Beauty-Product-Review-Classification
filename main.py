from bpReviewClassifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from bpReviewClassifier.pipeline.stage02_data_transformation import DataTransformationPipeline
from bpReviewClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from bpReviewClassifier.pipeline.stage_04_model_eval import ModelEvaluationPipeline
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
STAGE_NAME='Model Training Stage'
try:
    logger.info(f'Starting {STAGE_NAME}')
    model_trainer=ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f'{STAGE_NAME} completed successfully')
except Exception as e:
    raise e
STAGE_NAME = 'Model Evaluation'
try:
    logger.info(f'Starting {STAGE_NAME}')
    model_eval=ModelEvaluationPipeline()
    model_eval.main()
    logger.info(f'{STAGE_NAME} completed successfully')
except Exception as e:
    raise e