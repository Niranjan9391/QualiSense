"""
Django settings for qualisense_project project.
"""

from pathlib import Path
import os
import firebase_admin
from firebase_admin import credentials, firestore
import json

# -----------------------------------------
# BASE SETTINGS
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "unsafe-dev-key")
DEBUG = os.environ.get("DEBUG", "False") == "True"

ALLOWED_HOSTS = ["*"]

# Using signed cookies because we are NOT using SQL database
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

CSRF_TRUSTED_ORIGINS = [
    "https://qualisense.onrender.com",
    "https://*.railway.app",
]


# -----------------------------------------
# INSTALLED APPS
# -----------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "quali_defect_app",
]


# -----------------------------------------
# MIDDLEWARE
# -----------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]


# -----------------------------------------
# URL + WSGI
# -----------------------------------------
ROOT_URLCONF = "qualisense_project.urls"
WSGI_APPLICATION = "qualisense_project.wsgi.application"


# -----------------------------------------
# ❌ Disable SQL DATABASE (Firebase Only Mode)
# -----------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.dummy",  # Prevent Django ORM from being used
    }
}

# Disable Django password validators — Firebase Auth handles login
AUTH_PASSWORD_VALIDATORS = []


# -----------------------------------------
# TEMPLATES
# -----------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],  # Add global templates folder if needed
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "quali_defect_app.context_processors.firebase_keys",
            ],
        },
    },
]



# payment related keys
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")


# -----------------------------------------
# STATIC + MEDIA FILES
# -----------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"



MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# -----------------------------------------
# 🔥 FIREBASE INITIALIZATION
# -----------------------------------------
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")

if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(FIREBASE_CREDENTIALS_JSON)
    )
    firebase_admin.initialize_app(cred)

# Firestore client — used for storing all records
FIRESTORE_DB = firestore.client()


FIREBASE_API_KEY = "AIzaSyBPjTIFC8czCOm_q6wxSCRH3RkbXQ543gg"
FIREBASE_AUTH_DOMAIN = "qualisense-7ee06.firebaseapp.com"
FIREBASE_PROJECT_ID = "qualisense-7ee06"


# -----------------------------------------
# OTHER SETTINGS
# -----------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
