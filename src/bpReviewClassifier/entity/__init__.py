from dataclasses import dataclass,field
from pathlib import Path
from typing import List
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    input_data_file: Path
    tokenizer_name:Path
