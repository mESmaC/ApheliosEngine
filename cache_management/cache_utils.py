import redis
import json
import logging

logger = logging.getLogger("CacheUtils")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def set_cache(key, value, expiration=3600):
    """Set a value in the Redis cache with an optional expiration time."""
    try:
        redis_client.setex(key, expiration, json.dumps(value))
        logger.info(f"Set cache for key: {key}")
    except Exception as e:
        logger.error(f"Error setting cache for key {key}: {e}")

def get_cache(key):
    """Get a value from the Redis cache."""
    try:
        value = redis_client.get(key)
        if value:
            logger.info(f"Cache hit for key: {key}")
            return json.loads(value)
        logger.info(f"Cache miss for key: {key}")
        return None
    except Exception as e:
        logger.error(f"Error getting cache for key {key}: {e}")
        return None

def delete_cache(key):
    """Delete a value from the Redis cache."""
    try:
        redis_client.delete(key)
        logger.info(f"Deleted cache for key: {key}")
    except Exception as e:
        logger.error(f"Error deleting cache for key {key}: {e}")

def clear_cache():
    """Clear all values from the Redis cache."""
    try:
        redis_client.flushdb()
        logger.info("Cleared all cache")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
