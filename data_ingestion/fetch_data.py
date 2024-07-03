import logging
import threading
from utils.logger import setup_logger
from firebase_init import db  
import spacy
from cache_management.cache_utils import get_cache, set_cache

logger = setup_logger()

nlp = spacy.load("en_core_web_sm")

cache_lock = threading.Lock()

def fetch_new_data():
    try:
        with cache_lock:
            cached_data = get_cache('new_data')
            if cached_data:
                return cached_data

        users_ref = db.collection('users')
        videos_ref = db.collection('videos')

        users = users_ref.stream()

        data = []

        for user in users:
            user_data = user.to_dict()
            user_id = user.id
            user_interests = user_data.get('tags', [])
            watched_views = user_data.get('watchedViews', [])

            videos = videos_ref.stream()  

            for video in videos:
                video_data = video.to_dict()
                video_id = video.id
                video_tags = video_data.get('tags', [])
                video_description = video_data.get('description', '')
                video_views = video_data.get('views', 0)
                video_impressions = video_data.get('impressions', 0)
                video_likes = video_data.get('likes', [])
                video_comments_count = video_data.get('comcount', 0)

                comments_ref = videos_ref.document(video_id).collection('comments')
                comments = comments_ref.stream()
                comments_data = []
                for comment in comments:
                    comment_data = comment.to_dict()
                    comments_data.append({
                        "content": comment_data.get('content', ''),
                        "date": comment_data.get('date', None),
                        "dislikes": comment_data.get('dislikes', 0),
                        "likes": comment_data.get('likes', 0),
                        "user": comment_data.get('user', '')
                    })

                if '#2x10862CE' in video_tags:
                    continue

                data.append({
                    "user_id": user_id,
                    "interests": user_interests,
                    "watched_views": watched_views,
                    "video_id": video_id,
                    "tags": video_tags,
                    "description": video_description,
                    "retention": 0,  
                    "likes": video_likes,
                    "comments_count": video_comments_count,
                    "comments": comments_data,
                    "impressions": video_impressions,
                    "views": video_views
                })

        with cache_lock:
            set_cache('new_data', data)
        logger.info(f"Fetched {len(data)} records from Firebase.")
        return data
    except Exception as e:
        logger.error(f"Error fetching new data: {e}")
        return []
