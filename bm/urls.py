
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

from bm import settings
from bm.views import health

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
    #path('betse/', include("_betse.urls")),

    path('world/', include("urls")),
    path('auth/', include("auth.urls")),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

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