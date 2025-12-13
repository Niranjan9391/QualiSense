# quali_defect_app/firebase_utils.py
import firebase_admin
from firebase_admin import credentials, auth, firestore
from django.conf import settings
from django.contrib.auth import get_user_model, login as django_login, logout as django_logout
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from functools import wraps
from django.shortcuts import redirect
from django.http import JsonResponse

# -------------------------------------------------------------------
# SAFE FIREBASE INITIALIZATION (uses serviceAccountKey.json path from settings)
# -------------------------------------------------------------------
FIREBASE_CRED_PATH = getattr(settings, "FIREBASE_CRED_PATH", None)

if not firebase_admin._apps:
    if FIREBASE_CRED_PATH is None:
        raise RuntimeError("FIREBASE_CRED_PATH not set in settings.py")
    cred = credentials.Certificate(str(FIREBASE_CRED_PATH))
    firebase_admin.initialize_app(cred, {
        # optional extras:
        "projectId": getattr(settings, "FIREBASE_PROJECT_ID", None),
        "storageBucket": getattr(settings, "FIREBASE_STORAGE_BUCKET", None),
    })

# Firestore client (optional for data writes/reads)
db = firestore.client()

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
SESSION_COOKIE_NAME = getattr(settings, "FIREBASE_SESSION_COOKIE_NAME", "firebase_session")
# how long session cookie lives (days) — configurable via settings if desired
SESSION_COOKIE_EXPIRES_DAYS = getattr(settings, "FIREBASE_SESSION_COOKIE_EXPIRES_DAYS", 5)

# -------------------------------------------------------------------
# Firebase helper functions
# -------------------------------------------------------------------
def create_firebase_user_email(email: str, password: str, display_name: str = None):
    """
    Create a Firebase Authentication user with email/password.
    Returns created firebase user record dict or raises firebase_admin.auth.AuthError.
    """
    try:
        user = auth.create_user(email=email, password=password, display_name=display_name)
        return user
    except Exception as e:
        # re-raise for caller to handle (message contains Firebase's explanation)
        raise

def get_firebase_user_by_email(email: str):
    try:
        return auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        return None

def get_firebase_user(uid: str):
    try:
        return auth.get_user(uid)
    except auth.UserNotFoundError:
        return None

def verify_id_token(id_token: str):
    """
    Verifies an ID token issued by Firebase (client side).
    Returns decoded token dict on success.
    """
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        raise

def create_session_cookie_from_id_token(id_token: str, expires_days: int = SESSION_COOKIE_EXPIRES_DAYS):
    """
    Create a Firebase session cookie from an ID token.
    Use this to create a server-side cookie that persists longer than an ID token.
    """
    expires = timedelta(days=expires_days)
    try:
        session_cookie = auth.create_session_cookie(id_token, expires=expires)
        return session_cookie
    except Exception as e:
        raise

def verify_session_cookie(session_cookie: str, check_revoked: bool = True):
    """
    Verify a Firebase session cookie (created with create_session_cookie).
    Returns decoded claims.
    """
    try:
        decoded = auth.verify_session_cookie(session_cookie, check_revoked=check_revoked)
        return decoded
    except Exception as e:
        raise

# -------------------------------------------------------------------
# Django + Firebase integration helpers
# -------------------------------------------------------------------
def _get_or_create_django_user_from_firebase(decoded_token):
    """
    Given decoded Firebase token (contains 'uid' and maybe 'email'/'name'),
    create or fetch a Django user and return it.
    Username will be firebase UID by default (so unique).
    We set email and first_name if available.
    """
    UserModel = get_user_model()
    uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    name = decoded_token.get("name") or decoded_token.get("displayName")

    if uid is None:
        return None

    # Try to find an existing Django user by a custom field (username==uid) OR by email
    try:
        user = UserModel.objects.get(username=uid)
    except UserModel.DoesNotExist:
        # fallback: if email exists, try matching by email
        if email:
            try:
                user = UserModel.objects.get(email__iexact=email)
                # if found, update username to uid for consistent mapping
                user.username = uid
                if name:
                    if " " in name:
                        user.first_name = name.split(" ", 1)[0]
                    else:
                        user.first_name = name
                user.save()
            except UserModel.DoesNotExist:
                # create new user with username=uid
                user = UserModel.objects.create_user(username=uid, email=email or "", password=None)
                if name:
                    if " " in name:
                        user.first_name = name.split(" ", 1)[0]
                    else:
                        user.first_name = name
                user.save()
        else:
            # no email; create user with username=uid
            user = UserModel.objects.create_user(username=uid, email="", password=None)
            if name:
                if " " in name:
                    user.first_name = name.split(" ", 1)[0]
                else:
                    user.first_name = name
            user.save()

    return user

def login_user_from_id_token(request, id_token: str, use_session_cookie: bool = True):
    """
    Verify id_token, create or fetch a Django User, log them in and optionally
    create a secure session cookie for persistent server-side session.
    Returns dict with user and firebase claims.
    """
    decoded = verify_id_token(id_token)
    if not decoded:
        raise ValueError("Invalid ID token")

    user = _get_or_create_django_user_from_firebase(decoded)
    if user is None:
        raise RuntimeError("Could not create or fetch Django user from Firebase token")

    # Log Django user in (so request.user works)
    user.backend = "django.contrib.auth.backends.ModelBackend"
    django_login(request, user)

    # optionally create session cookie for longer-lived session (and set cookie)
    session_cookie = None
    if use_session_cookie:
        session_cookie = create_session_cookie_from_id_token(id_token)
        # set cookie on response - but this util does not have a response object.
        # The view that calls this util should set the cookie in HttpResponse.
    return {"user": user, "firebase_claims": decoded, "session_cookie": session_cookie}

def logout_django_and_revoke_session(request, response=None):
    """
    Logout Django user and optionally revoke Firebase session cookie and clear cookie on response.
    If you want to revoke at Firebase: auth.revoke_refresh_tokens(uid) can be called.
    """
    # revoke backend Django session
    try:
        uid = None
        if request.user.is_authenticated:
            uid = request.user.username  # we stored uid in username
    except:
        uid = None

    django_logout(request)

    if response is not None:
        response.delete_cookie(SESSION_COOKIE_NAME)

    # If we have a uid we can revoke refresh tokens at Firebase (forces re-login)
    if uid:
        try:
            auth.revoke_refresh_tokens(uid)
        except Exception:
            pass

# -------------------------------------------------------------------
# Decorator to protect views (works with Django session OR Firebase session cookie)
# -------------------------------------------------------------------
def firebase_login_required(redirect_to="/login/"):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            # 1) if Django auth already has user, allow
            if request.user.is_authenticated:
                return view_func(request, *args, **kwargs)

            # 2) check for session cookie (firebase session cookie)
            session_cookie = request.COOKIES.get(SESSION_COOKIE_NAME)
            if session_cookie:
                try:
                    claims = verify_session_cookie(session_cookie, check_revoked=True)
                except Exception:
                    return redirect(redirect_to)
                # create/login Django user from claims (but don't create new session cookie)
                user = _get_or_create_django_user_from_firebase(claims)
                if user:
                    user.backend = "django.contrib.auth.backends.ModelBackend"
                    django_login(request, user)
                    return view_func(request, *args, **kwargs)

            # Not authenticated
            return redirect(redirect_to)

        return _wrapped
    return decorator

# -------------------------------------------------------------------
# Convenience view snippets you can copy into views.py (examples)
# -------------------------------------------------------------------
def firebase_login_view(request):
    """
    POST endpoint. Expects JSON or form with 'idToken' (Firebase client ID token).
    Creates a session cookie and logs in Django user.
    Example frontend flow:
      - signInWithPopup(...) -> getIdToken() -> POST idToken to this endpoint.
    Response: JSON with status and optional session cookie (set in HttpResponse).
    """
    from django.http import JsonResponse, HttpResponse
    import json

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    body = request.POST or json.loads(request.body.decode("utf-8") or "{}")
    id_token = body.get("idToken")
    if not id_token:
        return JsonResponse({"error": "idToken missing"}, status=400)

    try:
        # login user and obtain session cookie
        result = login_user_from_id_token(request, id_token, use_session_cookie=True)
        session_cookie = result.get("session_cookie")
        response = JsonResponse({"status": "ok"})
        if session_cookie:
            # set cookie flags - set secure=True in production (HTTPS)
            max_age = SESSION_COOKIE_EXPIRES_DAYS * 24 * 60 * 60
            response.set_cookie(
                SESSION_COOKIE_NAME,
                session_cookie,
                max_age=max_age,
                httponly=True,
                secure=not settings.DEBUG,  # secure cookie in production
                samesite="Lax",
            )
        return response
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

def firebase_logout_view(request):
    """
    Logout endpoint: deletes cookie and logs out Django user.
    """
    from django.http import JsonResponse, HttpResponse
    response = JsonResponse({"status": "ok"})
    logout_django_and_revoke_session(request, response=response)
    return response
