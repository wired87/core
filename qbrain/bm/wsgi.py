"""
WSGI config for bm project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""
import os


from django.core.wsgi import get_wsgi_application
if os.name == "nt":
    GOOGLE_APPLICATION_CREDENTIALS = r"C:\Users\wired\OneDrive\Desktop\BestBrain\_google\g_auth\aixr-401704-59fb7f12485c.json"
else:
    GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/BestBrain/_google/g_auth/aixr-401704-59fb7f12485c.json"
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_APPLICATION_CREDENTIALS)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bm.settings')

application = get_wsgi_application()
