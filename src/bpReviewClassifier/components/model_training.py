from transformers import  Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset,load_from_disk
from bpReviewClassifier.entity import ModelConfig
import torch
import os
class ModelTrainer:
    def __init__(self,config:ModelConfig):
        self.config=config
    def train(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        #data_collator=DataCollatorWithPadding(tokenizer)
        model_review_classifer=AutoModelForSequenceClassification.from_pretrained(self.config.model_ckpt,num_labels=5)
       
        transformed_ds=load_from_disk(self.config.data_path)
       
        trainer_args=TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1, 
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy='steps',
            eval_steps=1000,
            save_steps=1e6,
            gradient_accumulation_steps=16,

            
        )
        trainer=Trainer(
            model=model_review_classifer,
            args=trainer_args,
            tokenizer=tokenizer,
            train_dataset=transformed_ds["test"],
            eval_dataset=transformed_ds["validation"]
        )
        trainer.train()
        model_review_classifer.save_pretrained(os.path.join(self.config.root_dir,"review-Classifier-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
                              
        