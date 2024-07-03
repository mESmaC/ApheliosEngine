import firebase_admin
from firebase_admin import credentials, firestore, app_check
from utils.config import FIREBASE_CREDENTIALS_PATH, FIRESTORE_PROJECT_ID, APP_CHECK_DEBUG_TOKEN

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        'projectId': FIRESTORE_PROJECT_ID,
    })

app_check.AppCheck.initialize_app_check(
    app_check.AppCheckOptions(
        debug_token=APP_CHECK_DEBUG_TOKEN
    )
)

db = firestore.client()