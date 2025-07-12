from django.urls import path

from auth.views.gh_access import GHAccess
from qf_sim.dj.views.demo import RunDemo
from qf_sim.dj.views.create_world import CreateWorldView

app_name = 'auth'
urlpatterns = [
    path('gh_access/', GHAccess.as_view()),
]
