from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("data-input/", views.data_input, name="data_input"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),

    # Auth
    path("login/", auth_views.LoginView.as_view(template_name="quali_defect_app/login.html"), name="login"),
    path("signup/", views.signup, name="signup"),
    path('logout/', views.logout_view, name='logout'),
]

