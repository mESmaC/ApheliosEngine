import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def set_cache(key, value, expiration=3600):
    """Set a value in the Redis cache with an optional expiration time."""
    redis_client.setex(key, expiration, json.dumps(value))

def get_cache(key):
    """Get a value from the Redis cache."""
    value = redis_client.get(key)
    if value:
        return json.loads(value)
    return None
