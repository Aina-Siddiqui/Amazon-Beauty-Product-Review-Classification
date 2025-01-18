import os
from pathlib import Path
from bpReviewClassifier.logging import logger
from ensure import ensure_annotations
import yaml
from box.exceptions import BoxValueError
from box import ConfigBox
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """
    Reads a YAML file and converts it into a ConfigBox object.
    Args:
        path_to_yaml (Path): Path to the YAML file.
    Raises:
        ValueError: If yaml file is empty
    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(config)
    except BoxValueError:
        raise ValueError('Empty YAML file {path_to_yaml}')
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    """Create directories
    Args:
      path_to_directories (list): List of paths to directories to be created.
       ignore_log(bool,optional):ignore if multiple directories is to be created : default to False
    """
    for directory in path_to_directories:
        os.makedirs(directory,exist_ok=True)
        if verbose:
                logger.info(f"{directory} Created successfully.")
@ensure_annotations
def get_size(path:Path)->str:
     """
     returns the size of the file in KB
     Args:
        paath (Path): path to the  file
     Returns:
        str: size of the file in KB
     """
     size=round(os.path.getsize(path)/1024)
     return f"{size} KB"
