import sqlite3
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("DBUtils")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

db_lock = threading.Lock()

class SQLiteConnectionPool:
    def __init__(self, database, pool_size=5):
        self.database = database
        self.pool_size = pool_size
        self.pool = [self._create_connection() for _ in range(pool_size)]

    def _create_connection(self):
        return sqlite3.connect(self.database, check_same_thread=False)

    @contextmanager
    def get_connection(self):
        with db_lock:
            conn = self.pool.pop() if self.pool else self._create_connection()
        try:
            yield conn
        finally:
            with db_lock:
                self.pool.append(conn)

db_pool = SQLiteConnectionPool('data.db')

def execute_query(query, params=None):
    """Execute a query and return the results."""
    with db_pool.get_connection() as conn:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None

def execute_non_query(query, params=None):
    """Execute a query that does not return results."""
    with db_pool.get_connection() as conn:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")

def create_table():
    """Create the data table if it doesn't exist."""
    query = '''
    CREATE TABLE IF NOT EXISTS data (
        user_id TEXT, 
        video_id TEXT, 
        interests TEXT, 
        tags TEXT, 
        description TEXT, 
        retention REAL, 
        likes INTEGER, 
        comments INTEGER, 
        correlate REAL
    )
    '''
    execute_non_query(query)

def insert_data(record):
    """Insert a record into the data table."""
    query = '''
    INSERT OR REPLACE INTO data (
        user_id, video_id, interests, tags, description, retention, likes, comments, correlate
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    params = (
        record["user_id"], 
        record["video_id"], 
        ','.join(record["interests"]), 
        ','.join(record["tags"]), 
        ' '.join(record["description"]), 
        record["retention"], 
        record["likes"], 
        record["comments"], 
        record["correlate"]
    )
    execute_non_query(query, params)
