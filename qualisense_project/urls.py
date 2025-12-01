from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", include("quali_defect_app.urls")),  # include your app routes
    # path("accounts/", include("django.contrib.auth.urls")), 
]
