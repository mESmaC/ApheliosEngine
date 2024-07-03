import threading
from database_management.db_utils import create_table, insert_data, execute_query

create_table()

db_lock = threading.Lock()

def update_database(data):
    """Update the SQLite database with new data."""
    with db_lock:
        for record in data:
            insert_data(record)

def get_total_impressions_and_views():
    """Get the total video impressions and views."""
    query = "SELECT SUM(impressions), SUM(views) FROM data"
    with db_lock:
        result = execute_query(query)
    if result:
        return result[0][0], result[0][1]
    return 0, 0
