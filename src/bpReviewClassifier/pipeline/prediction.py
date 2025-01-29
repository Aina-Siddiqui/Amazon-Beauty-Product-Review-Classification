from bpReviewClassifier.config.configuartion import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_eval_config()
    def predict(self,review):
        
        tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        classifier=pipeline('sentiment-analysis',model=self.config.model_path,tokenizer=tokenizer)
        result=classifier(review)
        return result