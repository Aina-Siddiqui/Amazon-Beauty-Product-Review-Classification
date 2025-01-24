import os
import pandas as pd
from bpReviewClassifier.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk,Dataset,DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from bpReviewClassifier.entity import DataTransformationConfig
class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)
    def preparing_dataset(self):
        ohe=OneHotEncoder()
        #loading dataset and converting into pandas dataframe
        df=pd.read_json(self.config.input_data_file,lines=True)
        #dropping unnecessary columns
        df.drop(['title','images','asin','parent_asin','user_id','timestamp', 'helpful_vote', 'verified_purchase'],axis=1,inplace=True)
        #one hot encoding our output column that is rating
        # Example fix for dataset labels
        df['rating'] = df['rating']-1
        #splitting dataset into train,test,validation
        train_df,temp_df=train_test_split(df,test_size=0.1,random_state=42)
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
        return dataset_dict
    def tokenize_fun(self,example_batch):
        encodings=self.tokenizer(example_batch['text'], truncation=True, padding='max_length')
        return encodings
    def convert(self):
        prepared_dataset=self.preparing_dataset()
        transformed=prepared_dataset.map(self.tokenize_fun,batched=True)
        transformed=transformed.remove_columns(['text'])
        transformed=transformed.rename_column('rating','label')
        transformed.save_to_disk(os.path.join(self.config.root_dir,"transformed_dataset"))
    
    

        