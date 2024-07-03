import time
import threading
import random
from firebase_admin import firestore
from cache_management.cache_utils import get_cache, set_cache
from model_serving.inference import recommend, fetch_user_and_video_ids, load_model
from utils.logger import setup_logger
from utils.config import MODEL_DIR
from firebase_init import db  

logger = setup_logger()

model = load_model(MODEL_DIR)

cache_lock = threading.Lock()

def check_for_new_users():
    try:
        if model is None:
            logger.warning("Model not loaded. Skipping new user check.")
            return

        users_ref = db.collection('UserData')
        users = users_ref.stream()

        with cache_lock:
            cached_user_ids = get_cache('user_ids') or []

        new_user_ids = []

        for user in users:
            user_id = user.id
            if user_id not in cached_user_ids:
                new_user_ids.append(user_id)
                user_ref = db.collection('UserData').document(user_id)
                algs_ref = user_ref.collection('algs').document('discover')
                if not algs_ref.get().exists:
                    algs_ref.set({'vid': []})

                _, video_ids = fetch_user_and_video_ids()

                if video_ids:
                    recommendations = recommend(user_id, model, video_ids, top_k=100)

                    if len(recommendations) < 100:
                        remaining_slots = 100 - len(recommendations)
                        random_videos = random.sample(video_ids, min(remaining_slots, len(video_ids)))
                        recommendations.extend(random_videos)

                    if len(recommendations) < 100:
                        logger.warning(f"Could only find {len(recommendations)} videos for user {user_id}")

                    algs_ref.update({
                        'vid': firestore.ArrayUnion(recommendations)
                    })
                else:
                    logger.warning(f"No videos found to recommend for user {user_id}")

        with cache_lock:
            cached_user_ids.extend(new_user_ids)
            set_cache('user_ids', cached_user_ids)

    except Exception as e:
        logger.error(f"Error checking for new users: {e}")

def start_background_task():
    def background_task():
        while True:
            check_for_new_users()
            time.sleep(10)  

    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
