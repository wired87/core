from django.contrib import admin
from django.urls import path, include, re_path
from django.conf.urls.static import static

from qbrain.bm import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    #path('health/', health),
    path('world/', include("qbrain.urls")),
    path('auth/', include("qbrain.auth.urls")),
    path('graph/', include("qbrain.graph.dj.urls")),
    path('api/relay/', include("qbrain.bm.api.relay")),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + [

]
