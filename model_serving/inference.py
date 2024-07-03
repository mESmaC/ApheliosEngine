import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import os
import threading
from utils.config import MODEL_DIR
from firebase_init import db  
from utils.logger import setup_logger

logger = setup_logger()

class MyModel(tfrs.Model):
    def __init__(self, user_model: tf.keras.Model, video_model: tf.keras.Model, task: tfrs.tasks.Retrieval):
        super().__init__()
        self.user_model = user_model
        self.video_model = video_model
        self.task = task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        video_embeddings = self.video_model(features["video_id"])
        return self.task(user_embeddings, video_embeddings)

def load_model(model_dir: str) -> MyModel:
    """Load the existing model from the specified directory."""
    try:
        user_model = tf.saved_model.load(os.path.join(model_dir, 'user_model'))
        video_model = tf.saved_model.load(os.path.join(model_dir, 'video_model'))
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=video_model))
        return MyModel(user_model, video_model, task)
    except OSError as e:
        logger.warning(f"Model files not found in {model_dir}. Please train the model first.")
        return None

def recommend(user_id: str, model: MyModel, video_ids: list, top_k: int = 10):
    """Generate recommendations for a given user."""
    user_embedding = model.user_model(np.array([user_id]))
    video_embeddings = model.video_model(np.array(video_ids))

    scores = tf.linalg.matmul(user_embedding, video_embeddings, transpose_b=True)
    top_k_indices = tf.math.top_k(scores, k=top_k).indices.numpy().flatten()

    recommended_video_ids = [video_ids[i] for i in top_k_indices]
    return recommended_video_ids

data_lock = threading.Lock()

def fetch_user_and_video_ids():
    """Fetch user and video IDs from Firebase."""
    with data_lock:
        users_ref = db.collection('users')
        videos_ref = db.collection('videos')

        user_ids = [user.id for user in users_ref.stream()]
        video_ids = [video.id for video in videos_ref.stream()]

    return user_ids, video_ids


if __name__ == "__main__":
    model = load_model(MODEL_DIR)

    user_ids, video_ids = fetch_user_and_video_ids()

    if user_ids and video_ids:
        user_id = user_ids[0]
        recommendations = recommend(user_id, model, video_ids)
        print(f"Recommendations for user {user_id}: {recommendations}")
    else:
        print("No users or videos found in Firebase.")