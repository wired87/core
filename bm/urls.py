
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('betse/', include("_betse.urls")),

    path('world/', include("qf_sim.dj.urls")),
]

