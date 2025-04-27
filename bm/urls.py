
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),


    path('betse/', include("_betse.urls")),
]

"""    path('client/', include("client.dashboard.urls.client")),
path('work/', include("client.dashboard.urls.admin")),
path('auth/', include("client.dashboard.auth.urls")),
path('gvs/', include("utils.ggoogle.vertexai.dj.urls")),
path('dai/', include("utils.ggoogle.documentai.views.urls")),
path('sp/', include("utils.ggoogle.spanner.dj.urls")),
path('queue/', include("utils.ggoogle.tasks.dj.urls")),
"""