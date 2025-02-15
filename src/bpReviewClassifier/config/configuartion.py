from bpReviewClassifier.constants import *
from bpReviewClassifier.utils.common import read_yaml,create_directories
from bpReviewClassifier.entity import DataIngestionConfig,DataTransformationConfig,ModelConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
    def get_data_ingestion(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    def get_data_transformation(self)->DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            input_data_file=config.input_data_file,
            tokenizer_name=config.tokenizer_filename
        )
        return data_transformation_config
    def get_model_config(self)->ModelConfig:
        config=self.config.model_trainer
        params=self.params.TrainingArguments
        create_directories([config.root_dir])
        model_trainer_config=ModelConfig(
        root_dir=config.root_dir,
        data_path=config.data_path,
        model_ckpt=config.model_ckpt,
        num_train_epochs=params.num_train_epochs,
        warmup_steps=params.warmup_steps,
        per_device_train_batch_size=params.per_device_train_batch_size,
        weight_decay=params.weight_decay,
        logging_steps=params.logging_steps,
        evaluation_strategy=params.evaluation_strategy,
        eval_steps=params.eval_steps,
        save_steps=params.save_steps,
        gradient_accumulation_steps=params.gradient_accumulation_steps
        )
        return model_trainer_config
    def get_model_eval_config(self)->ModelEvaluationConfig:
        config=self.config.model_evaluation
        create_directories([config.root_dir])
        model_eval_config=ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            data_path=Path(config.data_path),
            tokenizer_path=config.tokenizer_path,
            metric_filename=config.metric_filename,
        )
        return model_eval_config