from django.contrib import admin
from django.urls import path, include, re_path
from django.conf.urls.static import static

from qbrain.bm import settings
from qbrain.bm.views import health, spa_index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
    path('world/', include("qbrain.urls")),
    path('auth/', include("qbrain.auth.urls")),
    path('graph/', include("qbrain.graph.dj.urls")),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + [
    re_path(r'^.*$', spa_index),  # qdash SPA catch-all
]

"""
#path('bq/', include("_bq_core.dj.urls")),
path('docai/', include("documentai.views.urls")),

path('gke/', include("gke_admin.urls")),
path('batch/', include("cloud_batch.urls")),
path('eval/', include("qf_sim.evaluation.dj.urls")),

# frontend
#path("frontend/", include("frontend.urls")),
path("unicorn/", include("django_unicorn.urls", namespace="django_unicorn")),

"""