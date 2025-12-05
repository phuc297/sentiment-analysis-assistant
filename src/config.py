import os
from dotenv import load_dotenv

load_dotenv()

MODEL_SAVE_PATH = ''
MODEL_NAME = os.getenv('MODEL_NAME')
SA_API_URL = os.getenv('SA_API_URL')
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"