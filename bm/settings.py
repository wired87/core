import sys
from pathlib import Path
import os
from datetime import timedelta
import logging
import resend
from bm.logging_custom import cpr
import dotenv

OS_NAME = os.name
print("OS_NAME", OS_NAME)

if OS_NAME == "nt":
    GOOGLE_APPLICATION_CREDENTIALS = r"C:\\Users\\wired\\OneDrive\\Desktop\\Projects\\BestBrain\\_google\\g_auth\\aixr-401704-59fb7f12485c.json"

else:
    GOOGLE_APPLICATION_CREDENTIALS = "/home/derbenedikt_sterra/BestBrain/_google/g_auth/aixr-401704-59fb7f12485c.json"


dotenv.load_dotenv()

# Set absolute path to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
cpr("HI!")

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-gq0rr+q=s$=lci8=r%whnubclama3db!wnl1gpmh!_z2x5u_i3'

resend.api_key = "re_XaUKeMuq_EzcpCLqdqEk2o1m2H3tdBbw9"

if os.name == "nt":
    DEBUG = True
    TEST_USER_ID = "rajtigesomnlhfyqzbvx"
    REQUEST_URL = "http://127.0.0.1:8000/"
    GCP_TOKEN =r"C:\Users\wired\OneDrive\Desktop\Projects\bm\ggoogle\g_auth\token.json"

    TEST_ENV_ID="env_bare_rajtigesomnlhfyqzbvx"
    ALLOWED_HOSTS = ["*"]
    WS_URL = "ws://127.0.0.1:8000/"
    FIREBASE_CREDENTIALS = r"C:\\Users\\wired\OneDrive\\Desktop\\Projects\\Brainmaster\\_google\\g_auth\\firebase_creds.json"

else:
    DEBUG = False
    TEST_USER_ID = "rajtigesomnlhfyqzbvx" # todo
    TEST_ENV_ID="env_bare_rajtigesomnlhfyqzbvx"
    REQUEST_URL = "https://bm2-1004568990634.asia-east1.run.app/"
    GCP_TOKEN="utils/ggoogle/g_auth/token.json"
    ALLOWED_HOSTS = ['bestbrain.tech', 'www.bestbrain.tech', "34.69.187.50", "6000-firebase-studio-1749574544448.cluster-jbb3mjctu5cbgsi6hwq6u4btwe.cloudworkstations.dev"]
    WS_URL = "wss://bestbrain.tech/"
    FIREBASE_CREDENTIALS = r"_google/g_auth/firebase_creds.json"

CORS_ALLOW_ALL_ORIGINS = True


os.environ.setdefault("FIREBASE_CREDENTIALS", FIREBASE_CREDENTIALS)


allowed_main_host=REQUEST_URL.replace("https:", "").replace("http:", "").replace("/", "").replace(":8000", "")

"""
SOMETHIN HER EPROTECTS GET REQUESTS = !DRF
##########################################
# FIREWALL
##########################################
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

CSRF_TRUSTED_ORIGINS = ['https://bestbrain.tech', 'https://www.bestbrain.tech']
# X-Frame Protection (clickjacking)
X_FRAME_OPTIONS = 'DENY'

# Content sniffing protection
SECURE_CONTENT_TYPE_NOSNIFF = True

# Browser cross-site protections
SECURE_BROWSER_XSS_FILTER = True

# HSTS - Force HTTPS for all future connections
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True"""








LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'DEBUG',  # or INFO
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',  # or INFO
    },
}


CORS_ALLOW_CREDENTIALS = True


INSTALLED_APPS = [
    "channels",
    "django_unicorn",
    "frontend",
    "qf_sim",
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    "corsheaders",
    'rest_framework.authtoken',
    'rest_framework_simplejwt.token_blacklist',
    "_betse",
]

ASGI_APPLICATION = "bm.asgi.application"

# todo db basiert env speichern. -> fetch - set außerhalb proj
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Ensure this is FIRST
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    #'bm.middleware.AllowedMethodsMiddleware',
]

ROOT_URLCONF = 'bm.urls'
"""
# Beispiel-Konfiguration für Nginx, um den Zugriff zu blockieren:
location ~ /\.env {
    deny all;
    return 404; # Oder 403, um weniger Informationen preiszugeben
}

location ~ /\.git {
    deny all;
    return 404;
}

"""



# JWT
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=10),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=12),
    'ROTATE_REFRESH_TOKENS': True,  # Issue a new refresh token with each refresh
    'BLACKLIST_AFTER_ROTATION': True,  # Blacklist old tokens after refreshing
}
REST_FRAMEWORK = {
    """    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',  # default deny unless explicitly allowed
    ],"""
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    #'UNAUTHENTICATED_USER': None,  # Disallow Django auto-login fallback
}

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}



#AUTH_USER_MODEL = 'dashboard.UserModel'


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates']
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'bm.wsgi.application'


# STRIPE
STRIPE_PROD_API_KEY = os.getenv("STRIPE_PROD_API_KEY")
STRIPE_TEST_API_KEY = os.getenv("STRIPE_TEST_API_KEY")

STRIPE_ENDPOINT_SECRET_TEST = os.getenv("STRIPE_ENDPOINT_SECRET_TEST")
STRIPE_ENDPOINT_SECRET_PRODUCTION = os.getenv("STRIPE_ENDPOINT_SECRET_PRODUCTION")


# APIFY
APIFY_API_KEY = os.getenv("APIFY_API_TOKEN")

# CELERY
CELERY_TASK_DEFAULT_QUEUE = 'default'
# Celery Konfigurationen
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# Logging-Konfiguration
CELERYD_HIJACK_ROOT_LOGGER = False
CELERYD_LOG_COLOR = False

CELERYD_LOG_LEVEL = 'DEBUG'
CELERYD_PREFETCH_MULTIPLIER = 1
CELERYD_CONCURRENCY = 1

CELERY_WORKER_REDIRECT_STDOUTS = True
CELERY_WORKER_REDIRECT_STDOUTS_LEVEL = 'INFO'

# REDIS
REDIS_PUBLIC_ENDPOINT = os.getenv('REDIS_PUBLIC_ENDPOINT')
REDIS_PORT = os.getenv('REDIS_PORT')
REDIS_DB_PASSWORD = os.getenv("REDIS_DB_PASSWORD")
REDIS_BW_DB_ID = os.getenv("REDIS_BW_DB_ID")

REDIS_BROKER_URL = f'redis://:{REDIS_DB_PASSWORD}@{REDIS_PUBLIC_ENDPOINT}:{REDIS_PORT}/{REDIS_BW_DB_ID}'


# DJANGO-Q
Q_CLUSTER = {
    'name': 'DjangoRedis',
    'workers': 4,
    'timeout': 600,
    'retry': 601,
    "max_attempts": 4,
    "save_limit": 0,
    "guard_cycle": 1,
    'queue_limit': 50,
    'bulk': 5,
    "poll": 0.2,
    'redis': {
        'host': REDIS_PUBLIC_ENDPOINT,  #
        'port': REDIS_PORT,
        'db': 0,
        'password': REDIS_DB_PASSWORD,
        'socket_timeout': None,
    }
}



# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "static_root")
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]

MEDIA_ROOT = os.path.join(BASE_DIR, "media")

