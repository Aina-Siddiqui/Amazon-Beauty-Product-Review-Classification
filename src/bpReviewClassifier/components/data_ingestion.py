import os
import gzip
import shutil
import urllib.request as request
import zipfile
from bpReviewClassifier.logging import logger
from bpReviewClassifier.utils.common import get_size
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from bpReviewClassifier.entity import DataIngestionConfig
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            filename,headers=request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f'file already exist of size {get_size(Path(self.config.local_data_file))}')
    def extract_zip_file(self):
        unzip_dir=self.config.unzip_dir
        os.makedirs(unzip_dir,exist_ok=True)
        file_path = os.path.join(unzip_dir, 'BeautyProducts.jsonl')  
        with gzip.open(self.config.local_data_file, 'rb') as zip_ref:
             with open(file_path, 'wb') as decompressed_file:
                    shutil.copyfileobj(zip_ref, decompressed_file)
                    logger.info(f"GZ file extracted to {unzip_dir}")
        






        