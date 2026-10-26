from functools import wraps
import json
import os
import random
import re
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd
import mysql.connector

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from lime.lime_tabular import LimeTabularExplainer
from werkzeug.security import generate_password_hash, check_password_hash


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "dev-secret-key-change-me"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        connection_timeout=5,
        use_pure=True
    )


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "X_train.csv"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:
    best_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_train_for_lime = pd.read_csv(DATA_PATH)

    feature_names = X_train_for_lime.columns.tolist()

    # ----------------------------------------------------------
    # IMPORTANT: LIME is given the RAW (unscaled) training data,
    # not the scaled version. This means the thresholds LIME
    # reports later (e.g. "cholesterol above 260") will be in
    # real, human-readable units instead of meaningless scaled
    # decimals like 0.35. To make this work, we wrap the model's
    # predict_proba so it scales the input internally first —
    # LIME never needs to know scaling is happening.
    # ----------------------------------------------------------

    def scaled_predict_proba(raw_input):
        return best_model.predict_proba(
            scaler.transform(raw_input)
        )

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_for_lime.values,
        feature_names=feature_names,
        class_names=[
            "No Disease",
            "Disease"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    print("Machine learning model loaded successfully.")

except Exception as e:
    print("ML Model Loading Error:", e)
    best_model = None
    scaler = None
    lime_explainer = None
    feature_names = []
    scaled_predict_proba = None


# ============================================================
# HUMAN-FRIENDLY LIME TRANSLATION
# ============================================================
#
# LIME's raw output looks like: "254.00 < chol <= 293.00" or
# "sex > 0.50". This is accurate but not readable for a
# non-technical viewer (e.g. a professor during a demo).
# The dictionary and function below translate each condition
# into a plain-English sentence using real medical labels.

FEATURE_INFO = {
    "age": {
        "label": "Age",
        "unit": "years",
        "type": "continuous"
    },
    "sex": {
        "label": "Sex",
        "type": "categorical",
        "codes": {0: "Female", 1: "Male"}
    },
    "cp": {
        "label": "Chest Pain Type",
        "type": "categorical",
        "codes": {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "unit": "mmHg",
        "type": "continuous"
    },
    "chol": {
        "label": "Cholesterol Level",
        "unit": "mg/dL",
        "type": "continuous"
    },
    "fbs": {
        "label": "Fasting Blood Sugar Above 120 mg/dL",
        "type": "categorical",
        "codes": {0: "No", 1: "Yes"}
    },
    "restecg": {
        "label": "Resting ECG Result",
        "type": "categorical",
        "codes": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }
    },
    "thalach": {
        "label": "Maximum Heart Rate",
        "unit": "bpm",
        "type": "continuous"
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "type": "categorical",
        "codes": {0: "No", 1: "Yes"}
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "unit": "",
        "type": "continuous"
    },
    "slope": {
        "label": "ST Segment Slope",
        "type": "categorical",
        "codes": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    },
    "ca": {
        "label": "Number of Major Vessels",
        "type": "categorical",
        "codes": {0: "0", 1: "1", 2: "2", 3: "3"}
    },
    "thal": {
        "label": "Thalassemia Result",
        "type": "categorical",
        "codes": {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect"
        }
    }
}


def _format_number(key, value):
    if key == "oldpeak":
        return f"{value:.1f}"
    return f"{int(round(value))}"


def humanize_lime_condition(condition_text):
    """
    Converts a raw LIME condition string like:
        "254.00 < chol <= 293.00"
        "sex > 0.50"
        "cp <= 1.50"
    into a plain-English sentence using FEATURE_INFO.
    Falls back to the original text if parsing fails.
    """

    double_pattern = (
        r"^\s*([\d.\-]+)\s*(<=|<)\s*([a-zA-Z_]+)\s*"
        r"(<=|<)\s*([\d.\-]+)\s*$"
    )

    single_pattern = (
        r"^\s*([a-zA-Z_]+)\s*(<=|<|>=|>)\s*([\d.\-]+)\s*$"
    )

    key = None
    low = None
    low_op = None
    high = None
    high_op = None

    match = re.match(double_pattern, condition_text)

    if match:
        low = float(match.group(1))
        low_op = match.group(2)
        key = match.group(3)
        high_op = match.group(4)
        high = float(match.group(5))

    else:
        match = re.match(single_pattern, condition_text)

        if match:
            key = match.group(1)
            op = match.group(2)
            val = float(match.group(3))

            if op in (">", ">="):
                low = val
                low_op = "<" if op == ">" else "<="
            else:
                high = val
                high_op = op

    if key is None or key not in FEATURE_INFO:
        return condition_text

    info = FEATURE_INFO[key]
    label = info["label"]

    if info["type"] == "categorical":

        codes = info["codes"]
        matching_labels = []

        for code, code_label in codes.items():

            ok = True

            if low is not None:
                if low_op == "<" and not (low < code):
                    ok = False
                if low_op == "<=" and not (low <= code):
                    ok = False

            if high is not None:
                if high_op == "<" and not (code < high):
                    ok = False
                if high_op == "<=" and not (code <= high):
                    ok = False

            if ok:
                matching_labels.append(code_label)

        if not matching_labels:
            return condition_text

        if len(matching_labels) == 1:
            return f"{label} is {matching_labels[0]}"

        return f"{label} is " + " or ".join(matching_labels)

    unit = info.get("unit", "")
    unit_suffix = f" {unit}" if unit else ""

    if low is not None and high is not None:
        low_str = _format_number(key, low)
        high_str = _format_number(key, high)
        return (
            f"{label} is between {low_str} and "
            f"{high_str}{unit_suffix}"
        )

    if low is not None:
        low_str = _format_number(key, low)
        return f"{label} is above {low_str}{unit_suffix}"

    if high is not None:
        high_str = _format_number(key, high)
        return f"{label} is {high_str}{unit_suffix} or below"

    return condition_text


# ============================================================
# GMAIL CONFIGURATION
# ============================================================

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


# ============================================================
# PASSWORD RESET OTP STORAGE
# ============================================================

otp_storage = {}


# ============================================================
# SEND PASSWORD RESET OTP EMAIL
# ============================================================

def send_otp_email(receiver_email, otp):

    message = EmailMessage()

    message["Subject"] = (
        "Heart Risk Prediction System - Password Reset OTP"
    )

    message["From"] = GMAIL_ADDRESS
    message["To"] = receiver_email

    message.set_content(
        f"""
Hello,

Your password reset OTP is:

{otp}

This OTP is valid for 10 minutes.

If you did not request a password reset,
please ignore this email.

Regards,
Heart Risk Prediction System
"""
    )

    try:

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                GMAIL_ADDRESS,
                GMAIL_APP_PASSWORD
            )

            smtp.send_message(message)

        print(
            f"OTP email sent successfully to {receiver_email}"
        )

        return True

    except Exception as e:

        print("Email Error:", e)

        return False


# ============================================================
# SEND PREDICTION EMAIL NOTIFICATION
# ============================================================

def send_prediction_email(
    receiver_email,
    username,
    prediction,
    confidence_score,
    prediction_time
):

    if prediction == 1:
        risk_text = "Disease Predicted"
    else:
        risk_text = "No Disease Predicted"

    message = EmailMessage()

    message["Subject"] = (
        "Heart Risk Prediction System - Prediction Alert"
    )

    message["From"] = GMAIL_ADDRESS
    message["To"] = receiver_email

    message.set_content(
        f"""
Hello {username},

Your heart risk prediction has been completed successfully.

----------------------------------------
PREDICTION DETAILS
----------------------------------------

Prediction Result:
{risk_text}

Model Confidence:
{confidence_score:.2f}%

Date and Time:
{prediction_time}

----------------------------------------

IMPORTANT INFORMATION

This result was generated by a machine-learning
model for academic/project purposes.

The model prediction and confidence score should
not be interpreted as a medical diagnosis.

If you have health concerns, please consult a
qualified healthcare professional.

Regards,
Heart Risk Prediction System
"""
    )

    try:

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                GMAIL_ADDRESS,
                GMAIL_APP_PASSWORD
            )

            smtp.send_message(message)

        print(
            f"Prediction email sent successfully to {receiver_email}"
        )

        return True

    except Exception as e:

        print("Prediction Email Error:", e)

        return False


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login to continue.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        return f(*args, **kwargs)

    return decorated_function


# ============================================================
# ADMIN REQUIRED DECORATOR
# ============================================================

def admin_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login first.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        if session.get("role") != "admin":

            flash(
                "Admin access required.",
                "error"
            )

            return redirect(
                url_for("dashboard")
            )

        return f(*args, **kwargs)

    return decorated_function


# ============================================================
# HOME PAGE (LANDING PAGE)
# ============================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "landing.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not username or not email or not password:

            flash(
                "Please fill all required fields.",
                "error"
            )

            return render_template(
                "signup.html"
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "signup.html"
            )

        connection = None
        cursor = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor(
                dictionary=True
            )

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE username = %s
                   OR email = %s
                """,
                (
                    username,
                    email
                )
            )

            existing_user = cursor.fetchone()

            if existing_user:

                flash(
                    "Username or email already exists.",
                    "error"
                )

                return render_template(
                    "signup.html"
                )

            hashed_password = generate_password_hash(
                password
            )

            cursor.execute(
                """
                INSERT INTO users
                (
                    username,
                    email,
                    password,
                    role,
                    status
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    username,
                    email,
                    hashed_password,
                    "user",
                    "active"
                )
            )

            connection.commit()

            flash(
                "Account created successfully. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except mysql.connector.Error as e:

            print("Signup Database Error:", e)

            flash(
                "Unable to create account.",
                "error"
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        connection = None
        cursor = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor(
                dictionary=True
            )

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    email,
                    password,
                    role,
                    status
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            user = cursor.fetchone()

            if not user:

                flash(
                    "Invalid email or password.",
                    "error"
                )

                return render_template(
                    "login.html"
                )

            if user["status"] != "active":

                flash(
                    "Your account is not active.",
                    "error"
                )

                return render_template(
                    "login.html"
                )

            if not check_password_hash(
                user["password"],
                password
            ):

                flash(
                    "Invalid email or password.",
                    "error"
                )

                return render_template(
                    "login.html"
                )

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["email"] = user["email"]
            session["role"] = user["role"]

            if user["role"] == "admin":

                return redirect(
                    url_for("admin_dashboard")
                )

            return redirect(
                url_for("dashboard")
            )

        except mysql.connector.Error as e:

            print("Login Database Error:", e)

            flash(
                "Unable to login.",
                "error"
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route(
    "/forgot-password",
    methods=["GET", "POST"]
)
def forgot_password():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        connection = None
        cursor = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor(
                dictionary=True
            )

            cursor.execute(
                """
                SELECT id, username, email
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            user = cursor.fetchone()

            if not user:

                flash(
                    "No account found with this email.",
                    "error"
                )

                return render_template(
                    "forgot_password.html"
                )

            otp = str(
                random.randint(
                    100000,
                    999999
                )
            )

            expiry = (
                datetime.now()
                + timedelta(minutes=10)
            )

            otp_storage[email] = {
                "otp": otp,
                "expiry": expiry
            }

            email_sent = send_otp_email(
                email,
                otp
            )

            if email_sent:

                session["reset_email"] = email

                flash(
                    "OTP sent to your email.",
                    "success"
                )

                return redirect(
                    url_for("verify_otp")
                )

            flash(
                "Unable to send OTP email.",
                "error"
            )

        except mysql.connector.Error as e:

            print(
                "Forgot Password Database Error:",
                e
            )

            flash(
                "Unable to process request.",
                "error"
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "forgot_password.html"
    )


# ============================================================
# VERIFY OTP
# ============================================================

@app.route(
    "/verify-otp",
    methods=["GET", "POST"]
)
def verify_otp():

    email = session.get(
        "reset_email"
    )

    if not email:

        flash(
            "Password reset session expired.",
            "error"
        )

        return redirect(
            url_for("forgot_password")
        )

    if request.method == "POST":

        entered_otp = request.form.get(
            "otp",
            ""
        ).strip()

        stored_data = otp_storage.get(
            email
        )

        if not stored_data:

            flash(
                "OTP expired. Please request a new OTP.",
                "error"
            )

            return redirect(
                url_for("forgot_password")
            )

        if datetime.now() > stored_data["expiry"]:

            otp_storage.pop(
                email,
                None
            )

            flash(
                "OTP expired. Please request a new OTP.",
                "error"
            )

            return redirect(
                url_for("forgot_password")
            )

        if entered_otp != stored_data["otp"]:

            flash(
                "Invalid OTP.",
                "error"
            )

            return render_template(
                "verify_otp.html"
            )

        session["otp_verified"] = True

        otp_storage.pop(
            email,
            None
        )

        return redirect(
            url_for("reset_password")
        )

    return render_template(
        "verify_otp.html"
    )


# ============================================================
# RESET PASSWORD
# ============================================================

@app.route(
    "/reset-password",
    methods=["GET", "POST"]
)
def reset_password():

    email = session.get(
        "reset_email"
    )

    if not email or not session.get(
        "otp_verified"
    ):

        flash(
            "Please verify OTP first.",
            "error"
        )

        return redirect(
            url_for("forgot_password")
        )

    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not password:

            flash(
                "Password cannot be empty.",
                "error"
            )

            return render_template(
                "reset_password.html"
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "reset_password.html"
            )

        hashed_password = generate_password_hash(
            password
        )

        connection = None
        cursor = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            cursor.execute(
                """
                UPDATE users
                SET password = %s
                WHERE email = %s
                """,
                (
                    hashed_password,
                    email
                )
            )

            connection.commit()

            session.pop(
                "reset_email",
                None
            )

            session.pop(
                "otp_verified",
                None
            )

            flash(
                "Password reset successfully. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except mysql.connector.Error as e:

            print(
                "Reset Password Database Error:",
                e
            )

            flash(
                "Unable to reset password.",
                "error"
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "reset_password.html"
    )


# ============================================================
# PROFILE
# ============================================================

@app.route(
    "/profile",
    methods=["GET", "POST"]
)
@login_required
def profile():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        user_id = session["user_id"]

        if request.method == "POST":

            username = request.form.get(
                "username",
                ""
            ).strip()

            email = request.form.get(
                "email",
                ""
            ).strip()

            if not username or not email:

                flash(
                    "Username and email cannot be empty.",
                    "error"
                )

                return redirect(
                    url_for("profile")
                )

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE (username = %s OR email = %s)
                  AND id != %s
                """,
                (
                    username,
                    email,
                    user_id
                )
            )

            existing_user = cursor.fetchone()

            if existing_user:

                flash(
                    "Username or email is already in use.",
                    "error"
                )

                return redirect(
                    url_for("profile")
                )

            cursor.execute(
                """
                UPDATE users
                SET username = %s,
                    email = %s
                WHERE id = %s
                """,
                (
                    username,
                    email,
                    user_id
                )
            )

            connection.commit()

            session["username"] = username
            session["email"] = email

            flash(
                "Profile updated successfully.",
                "success"
            )

            return redirect(
                url_for("profile")
            )

        cursor.execute(
            """
            SELECT
                id,
                username,
                email,
                role,
                status,
                created_at
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user = cursor.fetchone()

        return render_template(
            "profile.html",
            user=user
        )

    except mysql.connector.Error as e:

        print(
            "Profile Database Error:",
            e
        )

        flash(
            "Unable to load profile.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# CHANGE PASSWORD
# ============================================================

@app.route(
    "/change-password",
    methods=["POST"]
)
@login_required
def change_password():

    current_password = request.form.get(
        "current_password",
        ""
    )

    new_password = request.form.get(
        "new_password",
        ""
    )

    confirm_password = request.form.get(
        "confirm_password",
        ""
    )

    if new_password != confirm_password:

        flash(
            "New passwords do not match.",
            "error"
        )

        return redirect(
            url_for("profile")
        )

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT password
            FROM users
            WHERE id = %s
            """,
            (session["user_id"],)
        )

        user = cursor.fetchone()

        if not user or not check_password_hash(
            user["password"],
            current_password
        ):

            flash(
                "Current password is incorrect.",
                "error"
            )

            return redirect(
                url_for("profile")
            )

        hashed_password = generate_password_hash(
            new_password
        )

        cursor.execute(
            """
            UPDATE users
            SET password = %s
            WHERE id = %s
            """,
            (
                hashed_password,
                session["user_id"]
            )
        )

        connection.commit()

        flash(
            "Password changed successfully.",
            "success"
        )

        return redirect(
            url_for("profile")
        )

    except mysql.connector.Error as e:

        print(
            "Change Password Database Error:",
            e
        )

        flash(
            "Unable to change password.",
            "error"
        )

        return redirect(
            url_for("profile")
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
@login_required
def predict():

    if request.method == "GET":

        return render_template(
            "predict.html"
        )

    connection = None
    cursor = None

    try:

        # ----------------------------------------------------
        # GET INPUT VALUES
        # ----------------------------------------------------

        age = float(
            request.form["age"]
        )

        sex = float(
            request.form["sex"]
        )

        cp = float(
            request.form["cp"]
        )

        trestbps = float(
            request.form["trestbps"]
        )

        chol = float(
            request.form["chol"]
        )

        fbs = float(
            request.form["fbs"]
        )

        restecg = float(
            request.form["restecg"]
        )

        thalach = float(
            request.form["thalach"]
        )

        exang = float(
            request.form["exang"]
        )

        oldpeak = float(
            request.form["oldpeak"]
        )
        
        # ----------------------------------------------------
        # VALIDATE REALISTIC MEDICAL INPUT VALUES
        # ----------------------------------------------------

        # Reject NaN and infinity values
        values_to_check = {
            "Age": age,
            "Blood Pressure": trestbps,
            "Cholesterol": chol,
            "Maximum Heart Rate": thalach,
            "Oldpeak": oldpeak
        }

        for field_name, value in values_to_check.items():
            if not np.isfinite(value):
                flash(
                    f"{field_name} must be a valid finite number.",
                    "error"
                )
                return redirect(url_for("predict"))

        # Integer-only fields
        integer_fields = {
            "Age": age,
            "Blood Pressure": trestbps,
            "Cholesterol": chol,
            "Maximum Heart Rate": thalach
        }

        for field_name, value in integer_fields.items():
            if not value.is_integer():
                flash(
                    f"{field_name} must be a whole number.",
                    "error"
                )
                return redirect(url_for("predict"))

        # Allowed input ranges
        validation_rules = {
            "Age": (age, 1, 120),
            "Blood Pressure": (trestbps, 50, 300),
            "Cholesterol": (chol, 50, 1000),
            "Maximum Heart Rate": (thalach, 30, 250),
            "Oldpeak": (oldpeak, -10, 10)
        }

        for field_name, (value, minimum, maximum) in validation_rules.items():
            if not minimum <= value <= maximum:
                flash(
                    f"{field_name} must be between "
                    f"{minimum} and {maximum}.",
                    "error"
                )
                return redirect(url_for("predict"))


        slope = float(
            request.form["slope"]
        )

        ca = float(
            request.form["ca"]
        )

        thal = float(
            request.form["thal"]
        )


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        )


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if best_model is None or scaler is None:

            flash(
                "Machine learning model is not available.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = int(
            best_model.predict(
                input_scaled
            )[0]
        )


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            best_model,
            "predict_proba"
        ):

            probabilities = best_model.predict_proba(
                input_scaled
            )[0]

            confidence_score = float(
                np.max(probabilities) * 100
            )

        else:

            confidence_score = 0.0


        # ----------------------------------------------------
        # LIME EXPLANATION
        # ----------------------------------------------------

        top_factors = []

        if lime_explainer is not None:

            explanation = lime_explainer.explain_instance(
                input_data.values[0],
                scaled_predict_proba,
                num_features=8
            )

            for feature, weight in explanation.as_list():

                direction = (
                    "positive"
                    if weight > 0
                    else "negative"
                )

                friendly_text = humanize_lime_condition(
                    feature
                )

                effect_text = (
                    "increases risk"
                    if direction == "positive"
                    else "decreases risk"
                )

                top_factors.append(
                    {
                        "feature": friendly_text,
                        "effect": effect_text,
                        "weight": round(
                            float(weight),
                            4
                        ),
                        "direction": direction
                    }
                )


        # ----------------------------------------------------
        # CONVERT LIME DATA TO JSON
        # ----------------------------------------------------

        top_factors_json = json.dumps(
            top_factors
        )


        # ----------------------------------------------------
        # SAVE PREDICTION TO DATABASE
        # ----------------------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediction_logs
            (
                user_id,
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
                predicted_risk,
                confidence_score,
                top_factors,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                session["user_id"],
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
                prediction,
                confidence_score,
                top_factors_json,
                datetime.now()
            )
        )

        connection.commit()


        # ----------------------------------------------------
        # SEND EMAIL NOTIFICATION
        # ----------------------------------------------------

        prediction_time = datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )

        email_sent = send_prediction_email(
            receiver_email=session["email"],
            username=session["username"],
            prediction=prediction,
            confidence_score=confidence_score,
            prediction_time=prediction_time
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if prediction == 1:

            risk_text = "Disease Present"

        else:

            risk_text = "No Disease Detected"


        return render_template(
            "result.html",
            prediction=prediction,
            risk_text=risk_text,
            confidence_score=confidence_score,
            top_factors=top_factors,
            email_sent=email_sent
        )


    except KeyError as e:

        print(
            "Prediction Missing Field:",
            e
        )

        flash(
            "Please fill in all prediction fields.",
            "error"
        )

        return redirect(
            url_for("predict")
        )


    except ValueError as e:

        print(
            "Prediction Value Error:",
            e
        )

        flash(
            "Please enter valid numeric values.",
            "error"
        )

        return redirect(
            url_for("predict")
        )


    except mysql.connector.Error as e:

        print(
            "Prediction Database Error:",
            e
        )

        flash(
            "Unable to save prediction.",
            "error"
        )

        return redirect(
            url_for("predict")
        )


    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        flash(
            "An error occurred while processing the prediction.",
            "error"
        )

        return redirect(
            url_for("predict")
        )


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# REPORTS / PREDICTION HISTORY
# ============================================================

@app.route("/reports", methods=["GET"])
@login_required
def reports():
    connection = None
    cursor = None

    try:
        user_id = session["user_id"]

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id,
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
                predicted_risk,
                confidence_score,
                top_factors,
                created_at
            FROM prediction_logs
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))

        predictions = cursor.fetchall()

        # Convert LIME JSON text into Python list
        for prediction in predictions:

            if prediction.get("top_factors"):
                try:
                    prediction["top_factors"] = json.loads(
                        prediction["top_factors"]
                    )
                except Exception as e:
                    print("LIME JSON Error:", e)
                    prediction["top_factors"] = []
            else:
                prediction["top_factors"] = []

        return render_template(
            "reports.html",
            predictions=predictions
        )

    except mysql.connector.Error as e:

        print("Reports Database Error:", e)

        flash(
            "Unable to load prediction history.",
            "error"
        )

        return redirect(url_for("dashboard"))

    except Exception as e:

        print("Reports Error:", e)

        flash(
            "An error occurred while loading prediction history.",
            "error"
        )

        return redirect(url_for("dashboard"))

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    connection = None
    cursor = None

    try:

        user_id = session["user_id"]

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # TOTAL PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM prediction_logs
            WHERE user_id = %s
            """,
            (user_id,)
        )

        total_predictions = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM prediction_logs
            WHERE user_id = %s
              AND predicted_risk = 1
            """,
            (user_id,)
        )

        high_risk = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM prediction_logs
            WHERE user_id = %s
              AND predicted_risk = 0
            """,
            (user_id,)
        )

        low_risk = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # AVERAGE CONFIDENCE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT AVG(confidence_score)
            AS average_confidence
            FROM prediction_logs
            WHERE user_id = %s
            """,
            (user_id,)
        )

        result = cursor.fetchone()

        average_confidence = result[
            "average_confidence"
        ]

        if average_confidence is None:

            average_confidence = 0

        else:

            average_confidence = round(
                float(average_confidence),
                2
            )


        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                predicted_risk,
                confidence_score,
                created_at
            FROM prediction_logs
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (user_id,)
        )

        recent_predictions = cursor.fetchall()


        return render_template(
            "dashboard.html",
            username=session["username"],
            email=session["email"],
            total_predictions=total_predictions,
            high_risk=high_risk,
            low_risk=low_risk,
            average_confidence=average_confidence,
            recent_predictions=recent_predictions
        )


    except mysql.connector.Error as e:

        print(
            "Dashboard Database Error:",
            e
        )

        flash(
            "Unable to load dashboard information.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # TOTAL USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM users
            WHERE role = 'user'
            """
        )

        total_users = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # ACTIVE USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM users
            WHERE role = 'user'
              AND status = 'active'
            """
        )

        active_users = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # INACTIVE USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM users
            WHERE role = 'user'
              AND status = 'inactive'
            """
        )

        inactive_users = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # TOTAL PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM prediction_logs
            """
        )

        total_predictions = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # HIGH-RISK PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM prediction_logs
            WHERE predicted_risk = 1
            """
        )

        high_risk_predictions = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # RECENT USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                username,
                email,
                role,
                status,
                created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT 10
            """
        )

        users = cursor.fetchall()


        return render_template(
            "admin_dashboard.html",
            total_users=total_users,
            active_users=active_users,
            inactive_users=inactive_users,
            total_predictions=total_predictions,
            high_risk_predictions=high_risk_predictions,
            users=users
        )


    except mysql.connector.Error as e:

        print(
            "Admin Dashboard Database Error:",
            e
        )

        flash(
            "Unable to load admin dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# ADMIN - VIEW USERS
# ============================================================

# ============================================================
# ADMIN - VIEW USERS
# (redirects to admin_dashboard, which already shows the full
#  user table — admin_users.html doesn't exist as a separate page)
# ============================================================

@app.route("/admin/users")
@admin_required
def admin_users():

    return redirect(
        url_for("admin_dashboard")
    )


# ============================================================
# ADMIN - ADD USER   (NEWLY ADDED — fixes the BuildError)
# ============================================================

@app.route("/admin/user/add", methods=["GET", "POST"])
@admin_required
def add_user():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "user")
        status = request.form.get("status", "active")

        if not username or not email or not password:
            flash("Please fill all required fields.", "error")
            return render_template("add_user.html")

        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute(
                "SELECT id FROM users WHERE username = %s OR email = %s",
                (username, email)
            )
            existing_user = cursor.fetchone()

            if existing_user:
                flash("Username or email already exists.", "error")
                return render_template("add_user.html")

            hashed_password = generate_password_hash(password)

            cursor.execute(
                """
                INSERT INTO users (username, email, password, role, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (username, email, hashed_password, role, status)
            )
            connection.commit()

            flash("User added successfully.", "success")
            return redirect(url_for("admin_dashboard"))

        except mysql.connector.Error as e:
            print("Add User Database Error:", e)
            flash("Unable to add user.", "error")
            return render_template("add_user.html")

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    return render_template("add_user.html")


# ============================================================
# ADMIN - EDIT USER   (NEWLY ADDED — fixes the BuildError)
# ============================================================

@app.route("/admin/user/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_user(user_id):

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        if request.method == "POST":

            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            role = request.form.get("role", "user")
            status = request.form.get("status", "active")
            new_password = request.form.get("password", "").strip()

            if not username or not email:
                flash("Username and email cannot be empty.", "error")
                return redirect(url_for("edit_user", user_id=user_id))

            if new_password:
                hashed_password = generate_password_hash(new_password)
                cursor.execute(
                    """
                    UPDATE users
                    SET username = %s, email = %s, role = %s,
                        status = %s, password = %s
                    WHERE id = %s AND role != 'admin'
                    """,
                    (username, email, role, status, hashed_password, user_id)
                )
            else:
                cursor.execute(
                    """
                    UPDATE users
                    SET username = %s, email = %s, role = %s, status = %s
                    WHERE id = %s AND role != 'admin'
                    """,
                    (username, email, role, status, user_id)
                )
            connection.commit()

            flash("User updated successfully.", "success")
            return redirect(url_for("admin_dashboard"))

        cursor.execute(
            "SELECT id, username, email, role, status FROM users WHERE id = %s",
            (user_id,)
        )
        user = cursor.fetchone()

        if not user:
            flash("User not found.", "error")
            return redirect(url_for("admin_dashboard"))

        return render_template("edit_user.html", user=user)

    except mysql.connector.Error as e:
        print("Edit User Database Error:", e)
        flash("Unable to edit user.", "error")
        return redirect(url_for("admin_dashboard"))

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# ============================================================
# ADMIN - CHANGE USER STATUS
# ============================================================

@app.route(
    "/admin/user/<int:user_id>/status",
    methods=["GET", "POST"]
)
@admin_required
def toggle_user_status(user_id):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            "SELECT status FROM users WHERE id = %s",
            (user_id,)
        )

        user = cursor.fetchone()

        if not user:

            flash(
                "User not found.",
                "error"
            )

            return redirect(
                url_for("admin_dashboard")
            )

        new_status = (
            "inactive" if user["status"] == "active" else "active"
        )

        cursor.execute(
            """
            UPDATE users
            SET status = %s
            WHERE id = %s
              AND role != 'admin'
            """,
            (
                new_status,
                user_id
            )
        )

        connection.commit()

        flash(
            "User status updated successfully.",
            "success"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    except mysql.connector.Error as e:

        print(
            "Change User Status Error:",
            e
        )

        flash(
            "Unable to update user status.",
            "error"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# ADMIN - DELETE USER
# ============================================================

@app.route(
    "/admin/user/<int:user_id>/delete",
    methods=["POST"]
)
@admin_required
def delete_user(user_id):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        # Delete prediction history first
        cursor.execute(
            """
            DELETE FROM prediction_logs
            WHERE user_id = %s
            """,
            (user_id,)
        )

        # Delete user
        cursor.execute(
            """
            DELETE FROM users
            WHERE id = %s
              AND role != 'admin'
            """,
            (user_id,)
        )

        connection.commit()

        flash(
            "User deleted successfully.",
            "success"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    except mysql.connector.Error as e:

        print(
            "Delete User Error:",
            e
        )

        flash(
            "Unable to delete user.",
            "error"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# ADMIN - ALL PREDICTIONS
# ============================================================

@app.route("/admin/predictions")
@admin_required
def admin_predictions():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                p.id,
                u.username,
                u.email,
                p.predicted_risk,
                p.confidence_score,
                p.created_at
            FROM prediction_logs p
            JOIN users u
                ON p.user_id = u.id
            ORDER BY p.created_at DESC
            """
        )

        predictions = cursor.fetchall()

        return render_template(
            "admin_predictions.html",
            predictions=predictions
        )

    except mysql.connector.Error as e:

        print(
            "Admin Predictions Database Error:",
            e
        )

        flash(
            "Unable to load predictions.",
            "error"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# APPLICATION ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("----------------------------------------")
    print("Heart Risk Prediction System")
    print("----------------------------------------")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("----------------------------------------")

    app.run(
        debug=True
    )