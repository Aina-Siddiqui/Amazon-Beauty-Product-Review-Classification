from bpReviewClassifier.config.configuartion import ConfigurationManager
from bpReviewClassifier.components.data_transformation import DataTransformation
from bpReviewClassifier.logging import logger

class DataTransformationPipeline():
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation()
        data_transformation=DataTransformation(config=data_transformation_config)
        #prepared_dataset=data_transformation.preparing_dataset()
        #transformed=prepared_dataset.map(data_transformation.tokenize_fun,batched=True)
        data_transformation.convert()
