import os
from pathlib import Path
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

# File structure for recommender system
file_list = [
    "app.py",
    "config.py",
    "logger_config.py",
    "requirements.txt",
    "recommender/__init__.py",
    "recommender/tfidf_recommender.py",
    "recommender/data_processor.py",
    "recommender/utils.py",
    "data/.gitkeep",
    "models/.gitkeep",
    "logs/.gitkeep"
]

for filepath in file_list:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
