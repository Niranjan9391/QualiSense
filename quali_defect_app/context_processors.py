from django.conf import settings

def firebase_keys(request):
    return {
        "FIREBASE_API_KEY": settings.FIREBASE_API_KEY,
        "FIREBASE_AUTH_DOMAIN": settings.FIREBASE_AUTH_DOMAIN,
        "FIREBASE_PROJECT_ID": settings.FIREBASE_PROJECT_ID,
    }
