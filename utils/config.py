import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
