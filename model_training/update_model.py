import threading
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text
import os
from utils.logger import setup_logger

logger = setup_logger()

class MyModel(tfrs.Model):
    def __init__(self, user_model: tf.keras.Model, video_model: tf.keras.Model, task: tfrs.tasks.Retrieval):
        super().__init__()
        self.user_model = user_model
        self.video_model = video_model
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        video_embeddings = self.video_model(features["video_id"])
        return self.task(user_embeddings, video_embeddings)

def load_model(model_dir: str) -> MyModel:
    """Load the existing model from the specified directory."""
    user_model_path = os.path.join(model_dir, 'user_model')
    video_model_path = os.path.join(model_dir, 'video_model')

    try:
        user_model = tf.saved_model.load(user_model_path)
        video_model = tf.saved_model.load(video_model_path)
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=video_model))
        return MyModel(user_model, video_model, task)
    except OSError as e:
        logger.warning(f"Model files not found in {model_dir}. Please train the model first.")
        return None

def save_model(model: MyModel, model_dir: str):
    """Save the model to the specified directory."""
    tf.saved_model.save(model.user_model, os.path.join(model_dir, 'user_model'))
    tf.saved_model.save(model.video_model, os.path.join(model_dir, 'video_model'))


data_lock = threading.Lock()

def update_model(data, model_dir: str):
    """Update the model with new data."""
    with data_lock:
        model = load_model(model_dir)

        user_ids = [record["user_id"] for record in data]
        video_ids = [record["video_id"] for record in data]
        interests = [record["interests"] for record in data]
        tags = [record["tags"] for record in data]
        descriptions = [" ".join(record["description"]) for record in data]
        retention = [record["retention"] for record in data]
        views = [record["views"] for record in data]
        impressions = [record["impressions"] for record in data]

    dataset = tf.data.Dataset.from_tensor_slices({
        "user_id": user_ids,
        "video_id": video_ids,
        "interests": interests,
        "tags": tags,
        "description": descriptions,
        "retention": retention,
        "views": views,
        "impressions": impressions
    }).batch(4096)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(dataset, epochs=3)

    save_model(model, model_dir)
