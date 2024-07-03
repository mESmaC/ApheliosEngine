import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_CREDENTIALS_PATH = os.path.join(BASE_DIR, '..', 'secret', 'starlight-965f4-firebase-adminsdk-6a50j-52af2d23d8.json')
FIRESTORE_PROJECT_ID = 'starlight-965f4'  
FIRESTORE_DATABASE_URL = f'https://firestore.googleapis.com/v1/projects/starlight-965f4/databases/(default)/documents'
APP_CHECK_DEBUG_TOKEN = '184A76FF-4F28-4E90-8D2A-4471FAAFCF4C'  

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

SQLITE_DB_PATH = 'data.db'

MODEL_DIR = os.path.join(os.getcwd(), 'models')  
EMBEDDING_DIMENSION = 64
BATCH_SIZE = 4096
EPOCHS = 3
LEARNING_RATE = 0.1

LOG_FILE_PATH = 'recommendation_system.log'
LOG_LEVEL = 'INFO'

SCHEDULE_TIMES = []

for hour in range(24):
    for minute in [0, 30]:
        SCHEDULE_TIMES.append(f"{hour:02}:{minute:02}")

CACHE_EXPIRATION = 3600  