"""
WSGI config for bm project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""
import os

r"""if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    if os.name == "nt":
        GOOGLE_APPLICATION_CREDENTIALS = r"C:\\Users\\wired\\OneDrive\\Desktop\\Projects\\bm\utils\ggoogle\\g_auth\\aixr-401704-59fb7f12485c.json"
    else:
        GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/bm/utils/ggoogle/g_auth/aixr-401704-59fb7f12485c.json"

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_APPLICATION_CREDENTIALS)
"""
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bm.settings')


application = get_wsgi_application()
