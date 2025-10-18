from django.urls import path

from views.demo import RunDemo

app_name = 'world'
urlpatterns = [
    path('demo/', RunDemo.as_view()),
]
