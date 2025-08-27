
import dotenv
dotenv.load_dotenv()

init_in_local_project =f"""

gke-gcloud-auth-plugin --version

"""

INSTALL_AUTH_PLUGIN="""gcloud
components
install
gke - gcloud - auth - plugin"""


CREATE_PRIVATE_KEY = None
def get_priv_key(PRIVATE_KEY_FILE):
    return f"""
PRIVATE_KEY_FILE="/tmp/ec_private.pem"
openssl ecparam -genkey -name prime256v1 -noout -out ${PRIVATE_KEY_FILE}
"""


def cget_creds():
    return f"""
    gcloud container clusters get-credentials autopilot-cluster-1 --region us-central1 --project aixr-401704
    """