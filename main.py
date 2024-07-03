import sys
import os
import time
import schedule
import threading
import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
from data_ingestion.fetch_data import fetch_new_data
from data_ingestion.preprocess_data import preprocess_data
from model_training.train_model import train_model
from model_serving.inference import load_model, recommend, fetch_user_and_video_ids
from cache_management.cache_utils import set_cache, get_cache
from database_management.sqlite_db import update_database
from utils.logger import setup_logger
from utils.config import MODEL_DIR, SCHEDULE_TIMES, BATCH_SIZE
from data_ingestion.check_new_user import check_for_new_users, start_background_task
from firebase_init import db  

sys.path.insert(0, MODEL_DIR)

logger = setup_logger()

processed_data = []

data_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=5)

shutdown_event = threading.Event()

def model_exists(model_dir):
    user_model_path = os.path.join(model_dir, 'user_model', 'saved_model.pb')
    video_model_path = os.path.join(model_dir, 'video_model', 'saved_model.pb')
    return os.path.exists(user_model_path) and os.path.exists(video_model_path)

async def core_loop():
    global processed_data
    try:
        logger.info("Starting new iteration of core loop.")
        
        with data_lock:
            preprocessed_data = await asyncio.get_event_loop().run_in_executor(executor, preprocess_data, processed_data)
        
        model = await asyncio.get_event_loop().run_in_executor(executor, train_model, preprocessed_data)
        
        await asyncio.get_event_loop().run_in_executor(executor, set_cache, 'model', model)
        await asyncio.get_event_loop().run_in_executor(executor, update_database, preprocessed_data)
        
        logger.info("Iteration complete.")
    except Exception as e:
        logger.error(f"Error in core loop: {e}", exc_info=True)

async def fetch_data():
    global processed_data
    try:
        logger.info("Fetching new data from Firebase.")
        
        new_data = await asyncio.get_event_loop().run_in_executor(executor, fetch_new_data)
        preprocessed_data = await asyncio.get_event_loop().run_in_executor(executor, preprocess_data, new_data)
        
        with data_lock:
            processed_data.extend(preprocessed_data)
        
        logger.info("Data fetching complete.")
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)

async def write_to_firebase():
    global processed_data
    if processed_data:
        try:
            logger.info("Writing processed data to Firebase.")
            
            model = await asyncio.get_event_loop().run_in_executor(executor, load_model, MODEL_DIR)
            if model is None:
                logger.warning("Model not loaded. Skipping write to Firebase.")
                return
            
            user_ids, video_ids = await asyncio.get_event_loop().run_in_executor(executor, fetch_user_and_video_ids)

            if not user_ids or not video_ids:
                logger.error("No users or videos found in Firebase.")
                return

            batch_size = BATCH_SIZE
            with data_lock:
                for i in range(0, len(processed_data), batch_size):
                    batch = processed_data[i:i + batch_size]
                    
                    for record in batch:
                        user_id = record["user_id"]
                        recommendations = await asyncio.get_event_loop().run_in_executor(executor, recommend, user_id, model, video_ids)
                        
                        user_ref = db.collection('UserData').document(user_id)
                        user_data = user_ref.get().to_dict()
                        watched_views = user_data.get('watchedViews', [])

                        algs_ref = user_ref.collection('algs').document('discover')
                        current_recommendations = algs_ref.get().to_dict().get('vid', [])

                        updated_recommendations = [rec for rec in current_recommendations if rec not in watched_views]

                        updated_recommendations.extend(recommendations)

                        await asyncio.get_event_loop().run_in_executor(executor, algs_ref.update, {
                            'vid': updated_recommendations
                        })
                
                processed_data = []
        except Exception as e:
            logger.error(f"Error writing to Firebase: {e}", exc_info=True)
    else:
        logger.info("No data to write to Firebase.")

def schedule_tasks():
    for time_str in SCHEDULE_TIMES:
        schedule.every().day.at(time_str).do(lambda: asyncio.run(fetch_data()))
        schedule.every().day.at(time_str).do(lambda: asyncio.run(write_to_firebase()))
        schedule.every().day.at(time_str).do(lambda: asyncio.run(core_loop()))

    while not shutdown_event.is_set():
        try:
            schedule.run_pending()
            logger.info("Scheduler loop is running...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}", exc_info=True)

def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Shutting down gracefully...")
    shutdown_event.set()

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if model_exists(MODEL_DIR):
            scheduler_thread = threading.Thread(target=schedule_tasks)
            scheduler_thread.start()

            start_background_task()

            scheduler_thread.join()
        else:
            logger.warning("Model files not found. Please train the model first.")
            scheduler_thread = threading.Thread(target=schedule_tasks)
            scheduler_thread.start()

            start_background_task()

            scheduler_thread.join()
    except Exception as e:
        logger.error(f"Error starting the scheduler: {e}", exc_info=True)
