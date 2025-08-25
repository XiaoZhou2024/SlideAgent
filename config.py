import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY", "EMPTY")
    BASE_URL = os.getenv("BASE_URL", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "")

config = Config()
