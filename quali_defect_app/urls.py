from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("data-input/", views.data_input, name="data_input"),
    path("pricing/", views.pricing_page, name="pricing_page"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact_view, name="contact"),

    # 🔥 Firebase Auth Pages
    path("login/", views.login_page, name="login"),   # custom login page
    path("signup/", views.signup_page, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # 🔥 Firebase Login Session Endpoint
    path("session-login/", views.session_login, name="session_login"),

    # History + Export
    path("history/", views.history_view, name="history"),
    path('export/model1/excel/', views.export_model1_excel, name='export_model1_excel'),
    path('export/model2/excel/', views.export_model2_excel, name='export_model2_excel'),

    path("qs-admin/login/", views.admin_login_page, name="admin_login"),
    path("qs-admin/verify/", views.verify_admin_token, name="verify_admin"),
    path("qs-admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("qs-admin/users/", views.admin_users, name="admin_users"),
    path("qs-admin/logs/", views.admin_logs, name="admin_logs"),
    path("qs-admin/settings/", views.admin_settings, name="admin_settings"),
    
    #pricong related
    path("create-order/", views.create_order, name="create_order"),
    path("payment-success/", views.payment_success, name="payment_success"),
    path("api/user-credits/", views.user_credits_api, name="user_credits_api"),
    
    # Test
    path("test-firebase/", views.test_firebase),
]
