import threading
from flask import Flask, request, jsonify
from model_serving.inference import load_model, recommend, fetch_user_and_video_ids
from utils.config import MODEL_DIR
from firebase_init import db  
from firebase_admin import firestore  

app = Flask(__name__)


model = load_model(MODEL_DIR)

data_lock = threading.Lock()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    user_id = data.get('user_id')
    top_k = data.get('top_k', 10)

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400


    with data_lock:
        _, video_ids = fetch_user_and_video_ids()

    if not video_ids:
        return jsonify({"error": "No video IDs found in Firebase"}), 400

    user_ref = db.collection('UserData').document(user_id)
    user_data = user_ref.get().to_dict()
    watched_views = user_data.get('watchedViews', [])


    watched_video_ids = [view.rstrip('X') for view in watched_views]
    video_ids = [vid for vid in video_ids if vid not in watched_video_ids]

    recommendations = recommend(user_id, model, video_ids, top_k)

    try:
        algs_ref = user_ref.collection('algs').document('discover')


        if not algs_ref.get().exists:
            algs_ref.set({'vid': []})

        algs_ref.update({
            'vid': firestore.ArrayUnion(recommendations)
        })
        return jsonify({"message": "Recommendations updated successfully", "recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
