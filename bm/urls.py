
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

from bm import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    #path('betse/', include("_betse.urls")),

    path('world/', include("qf_sim.dj.urls")),
    path('eval/', include("qf_sim.evaluation.dj.urls")),
    path('auth/', include("auth.urls")),

    path('bq/', include("_bq_core.dj.urls")),

    # frontend
    path("frontend/", include("frontend.urls")),
    path("unicorn/", include("django_unicorn.urls", namespace="django_unicorn")),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

