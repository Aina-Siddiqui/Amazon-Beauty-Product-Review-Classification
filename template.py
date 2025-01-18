import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]''[%(message)s]')
project_name='bpReviewClassifier'
list_of_files =[
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuartion.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/logging/__init__.py',
    'app.py',
    'main.py',
    'setup.py',
    'requirements.txt',
    'research/trials.ipynb',
    'config/config.yaml',
    'params.yaml'
]
for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    if filedir!='':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir}')
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating file: {filepath}')
    else:
        logging.info(f'File {filepath} already exists')