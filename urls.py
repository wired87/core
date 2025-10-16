from django.urls import path

from views.create_node_cfg import CreateNodeCfgdView
from views.demo import RunDemo
from views.create_world import CreateWorldView

app_name = 'world'
urlpatterns = [
    path('create/', CreateWorldView.as_view()),
    path('create-ncfg/', CreateNodeCfgdView.as_view()),
    path('demo/', RunDemo.as_view()),
]
