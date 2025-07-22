from django.urls import path

from auth.views.get_creds_view import GetCredsView

app_name = 'auth'
urlpatterns = [
    path('access/', GetCredsView.as_view()),
]
