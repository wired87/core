from django.urls import path

from frontend.view import MyTemplateView

app_name = 'frontend'
urlpatterns = [
    path('main/', MyTemplateView.as_view()),
]
