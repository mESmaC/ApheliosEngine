import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import threading
from utils.config import MODEL_DIR
from utils.logger import setup_logger
import time

logger = setup_logger()

data_lock = threading.Lock()

def train_model(data):
    start_time = time.time()
    logger.info("Starting model training...")

    try:
        with data_lock:
            user_ids = [record["user_id"] for record in data]
            video_ids = [record["video_id"] for record in data]
            interests = [record["interests"] for record in data]
            tags = [record["tags"] for record in data]
            descriptions = [" ".join(record["description"]) for record in data]
            retention = [record["retention"] for record in data]
            views = [record["views"] for record in data]
            impressions = [record["impressions"] for record in data]
            likes = [len(record["likes"]) for record in data]  
            comments_count = [record["comments_count"] for record in data]  
            watched_views = [record["watched_views"] for record in data]  

        logger.info(f"Extracted {len(user_ids)} user_ids, {len(video_ids)} video_ids, and other features for training.")

    
        user_features = list(zip(retention, views, impressions, likes, comments_count))
        kmeans = KMeans(n_clusters=5, random_state=0).fit(user_features)
        user_clusters = kmeans.labels_

        dataset = tf.data.Dataset.from_tensor_slices({
            "user_id": user_ids,
            "video_id": video_ids,
            "interests": interests,
            "tags": tags,
            "description": descriptions,
            "retention": retention,
            "views": views,
            "impressions": impressions,
            "likes": likes,  
            "comments_count": comments_count, 
            "watched_views": watched_views,  
            "user_cluster": user_clusters  
        }).batch(4096).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_ids),
            tf.keras.layers.Embedding(input_dim=len(user_ids), output_dim=64)
        ])

        video_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=video_ids),
            tf.keras.layers.Embedding(input_dim=len(video_ids), output_dim=64)
        ])

        logger.info("Defined user and video models.")

        text_embedding_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tf.keras.layers.TextVectorization(max_tokens=20000, output_mode='int', output_sequence_length=200),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=64)
        ])

        logger.info("Defined text embedding model.")

        retrieval_model = tfrs.models.Model(
            user_model=user_model,
            item_model=video_model,
            task=tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=video_ids))
        )

        logger.info("Defined the retrieval model.")

        interaction_matrix = np.zeros((len(user_ids), len(video_ids)))
        for record in data:
            user_idx = user_ids.index(record["user_id"])
            video_idx = video_ids.index(record["video_id"])
            interaction_matrix[user_idx, video_idx] = record["views"]

        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(interaction_matrix)
        video_factors = svd.components_.T

        logger.info("Performed matrix factorization using SVD.")

        combined_user_model = tf.keras.layers.Concatenate()([user_model.output, user_factors])
        combined_video_model = tf.keras.layers.Concatenate()([video_model.output, video_factors])

        hybrid_model = tfrs.models.Model(
            user_model=tf.keras.Model(user_model.input, combined_user_model),
            item_model=tf.keras.Model(video_model.input, combined_video_model),
            task=tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=video_ids))
        )

        logger.info("Defined the hybrid recommendation model.")

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            hybrid_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
            logger.info("Compiled the hybrid recommendation model.")

            train_size = int(0.8 * len(data))
            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)

            tf.profiler.experimental.start('logdir')

            hybrid_model.fit(train_dataset, validation_data=val_dataset, epochs=3)
            logger.info("Model training complete.")

            tf.profiler.experimental.stop()

            eval_result = hybrid_model.evaluate(val_dataset)
            logger.info(f"Model evaluation result: {eval_result}")

            tf.saved_model.save(hybrid_model.user_model, os.path.join(MODEL_DIR, 'user_model'))
            tf.saved_model.save(hybrid_model.item_model, os.path.join(MODEL_DIR, 'video_model'))
            logger.info(f"Saved user and video models to {MODEL_DIR}.")

        end_time = time.time()
        logger.info(f"Model training completed in {end_time - start_time:.2f} seconds.")

        return hybrid_model

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise