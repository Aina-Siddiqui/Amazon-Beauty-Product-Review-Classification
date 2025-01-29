from bpReviewClassifier.components.model_evaluation import ModelEvaluation
from bpReviewClassifier.config.configuartion import ConfigurationManager
from bpReviewClassifier.logging import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_eval_config()
        model_evaluation=ModelEvaluation(config=model_evaluation_config)
        model_evaluation.loading_dataset()