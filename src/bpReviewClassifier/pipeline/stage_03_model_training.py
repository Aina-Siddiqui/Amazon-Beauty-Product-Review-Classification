from bpReviewClassifier.config.configuartion import ConfigurationManager
from bpReviewClassifier.components.model_training import ModelTrainer
from bpReviewClassifier.logging import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        model_training_config=config.get_model_config()
        model_trainer=ModelTrainer(config=model_training_config)
        model_trainer.train()