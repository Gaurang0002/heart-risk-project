import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import string

from app import app, get_db_connection


def random_email():
    suffix = "".join(random.choices(string.ascii_lowercase, k=8))
    return f"testuser_{suffix}@example.com"


def test_full_user_flow():
    client = app.test_client()
    test_email = random_email()
    test_username = test_email.split("@")[0]

    # ---- SIGNUP ----
    response = client.post("/signup", data={
        "username": test_username,
        "email": test_email,
        "password": "testpass123",
        "confirm_password": "testpass123"
    }, follow_redirects=True)
    assert response.status_code == 200

    # ---- LOGIN ----
    response = client.post("/login", data={
        "email": test_email,
        "password": "testpass123"
    }, follow_redirects=True)
    assert response.status_code == 200

    # ---- ACCESS PREDICT PAGE (should now be allowed) ----
    response = client.get("/predict")
    assert response.status_code == 200

    # ---- SUBMIT A PREDICTION ----
    response = client.post("/predict", data={
        "age": "55", "sex": "1", "cp": "2", "trestbps": "130",
        "chol": "250", "fbs": "0", "restecg": "1", "thalach": "150",
        "exang": "0", "oldpeak": "1.2", "slope": "1", "ca": "0", "thal": "2"
    }, follow_redirects=True)
    assert response.status_code == 200

    # ---- CHECK HISTORY SHOWS THE PREDICTION ----
    response = client.get("/reports")
    assert response.status_code == 200

    # ---- LOGOUT ----
    response = client.get("/logout", follow_redirects=True)
    assert response.status_code == 200

    # ---- CLEANUP: remove the test user and their data ----
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (test_email,))
    result = cursor.fetchone()
    if result:
        user_id = result[0]
        cursor.execute("DELETE FROM prediction_logs WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        connection.commit()
    cursor.close()
    connection.close()


def test_unauthenticated_predict_redirects_to_login():
    client = app.test_client()
    response = client.get("/predict", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.location