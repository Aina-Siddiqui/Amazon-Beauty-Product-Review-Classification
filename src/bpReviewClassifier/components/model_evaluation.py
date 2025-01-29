from transformers import  AutoModelForSequenceClassification,AutoTokenizer
from datasets import load_dataset,load_from_disk
from evaluate import load
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset,DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from bpReviewClassifier.entity import ModelEvaluationConfig
class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
    def tokenize_fun(self,batch):
            device='cuda' if torch.cuda.is_available() else 'cpu'
            tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_path)
    
            tokenized_inputs = tokenizer(batch['text'], truncation=True, padding='max_length',return_tensors='pt')
            return tokenized_inputs
    def collate_fn(self,batch):
                return {
                'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
                'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
            }
    def eval_acc(self,outputs,labels):
           
            dataloader = DataLoader(outputs, batch_size=32, collate_fn=self.collate_fn)
            model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
            all_preds = []
            total_batches = len(dataloader)  # Total number of batches

            for batch in tqdm(dataloader, total=total_batches, desc="Evaluating", unit="batch"):
                # Extract tensors from the batch
                input_ids = batch['input_ids']
            
                attention_mask = batch['attention_mask']

                with torch.no_grad():
                    preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
                
                all_preds.extend(preds.tolist())
               # Compute accuracy
            eval_acc = load('accuracy')
            score = eval_acc.compute(predictions=all_preds, references=labels)
            return score
    def loading_dataset(self):
        df=pd.read_json(self.config.data_path,lines=True)
        df.drop(['title','images','asin','parent_asin','user_id','timestamp', 'helpful_vote', 'verified_purchase'],axis=1,inplace=True)
        #one hot encoding our output column that is rating
        #df['rating']=df['rating']-1
        #splitting dataset into train,test,validation
        train_df,temp_df=train_test_split(df,test_size=0.3,random_state=42)
        test_df,val_df=train_test_split(temp_df,test_size=0.5,random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        #preparring datasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
            })
        
        dataset_dict=dataset_dict.rename_column('rating','label')
        labels=dataset_dict['test']['label']
        output=dataset_dict['test'].map(self.tokenize_fun,batched=True)
        

         
        result=self.eval_acc(output,labels)
        accuracy_dict=dict(accuracy=result)
        df=pd.DataFrame(accuracy_dict)
        df.to_csv(self.config.metric_filename,index=False)        
        
           