from flask import Flask, render_template, jsonify, request
import os
import threading
import asyncio
import signal
from main import fetch_data, core_loop, write_to_firebase, executor, shutdown_event
from utils.logger import setup_logger
from database_management.sqlite_db import get_total_impressions_and_views
from firebase_admin import app_check

logger = setup_logger()

app = Flask(__name__)

@app.before_request
def verify_app_check():
    app_check_token = request.headers.get('X-Firebase-AppCheck')
    if not app_check_token:
        return jsonify({"error": "AppCheck token missing"}), 403

    try:
        app_check.verify_token(app_check_token)
    except app_check.InvalidTokenError:
        return jsonify({"error": "Invalid AppCheck token"}), 403

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def get_logs():
    with open('recommendation_system.log', 'r') as file:
        logs = file.readlines()
    return jsonify(logs)

@app.route('/status')
def status():
    return jsonify({"status": "online"})

@app.route('/force_fetch', methods=['POST'])
def force_fetch():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info("Force fetch initiated.")
        loop.run_until_complete(fetch_data())
        loop.run_until_complete(core_loop())
        logger.info("Force fetch and training completed successfully.")
        return jsonify({"message": "Force fetch and training initiated."})
    except Exception as e:
        logger.error(f"Error during force fetch: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/force_write', methods=['POST'])
def force_write():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info("Force write initiated.")
        loop.run_until_complete(write_to_firebase())
        logger.info("Force write completed successfully.")
        return jsonify({"message": "Force write initiated."})
    except Exception as e:
        logger.error(f"Error during force write: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    try:
        logger.info("Safe shutdown initiated.")
        shutdown_event.set()
        os.kill(os.getpid(), signal.SIGTERM)  
        logger.info("Safe shutdown completed successfully.")
        return jsonify({"message": "Safe shutdown initiated."})
    except Exception as e:
        logger.error(f"Error during safe shutdown: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/video_stats')
def video_stats():
    try:
        total_impressions, total_views = get_total_impressions_and_views()
        return jsonify({"total_impressions": total_impressions, "total_views": total_views})
    except Exception as e:
        logger.error(f"Error fetching video stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)