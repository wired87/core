"""
ASGI config for bm project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import os

from qbrain.auth.set_gcp_auth_creds_path import set_gcp_auth_path
from qbrain.bm import routing

import dotenv
dotenv.load_dotenv()

set_gcp_auth_path()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'qbrain.bm.settings')

django_asgi_app = get_asgi_application()


# ============================================================================
# QBRAIN TABLE INITIALIZATION
# ============================================================================

def initialize_qbrain_tables():
    """
    Initialize QBRAIN dataset and tables on server startup.
    Runs once per server instance using TABLE_EXISTS environment variable.
    """
    table_exists_val = os.environ.get('TABLE_EXISTS', 'False').strip().lower()
    if table_exists_val not in ('true', 'false', '1', '0', ''):
        print(f"⚠ TABLE_EXISTS has invalid value '{os.environ.get('TABLE_EXISTS')}'; expected True/False")
    # Check if tables have already been initialized
    if table_exists_val in ('true', '1'):
        print("✓ QBRAIN tables already initialized (TABLE_EXISTS=True)")
        return
    
    try:
        # Use centralized table manager
        from qbrain.core.qbrain_manager import get_qbrain_table_manager

        print("\n🔧 Initializing QBRAIN Table Manager...")
        table_mgr = get_qbrain_table_manager()

        # Initialize all tables
        results = table_mgr.initialize_all_tables()

        # Set environment variable to prevent re-initialization
        os.environ['TABLE_EXISTS'] = 'True'

        print("✓ Environment variable set: TABLE_EXISTS=True\n")
    except Exception as e:
        print(f"\n❌ Error initializing QBRAIN tables: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ Server will continue but database operations may fail")
        print("=" * 70 + "\n")


# Run table initialization on server startup
initialize_qbrain_tables()


application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
            URLRouter(
                routing.websocket_urlpatterns
            )
        ),
    }
)
