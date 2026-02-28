import os

from google.cloud import compute_v1

def get_vm_public_ip_address(instance_name):
    import dotenv
    dotenv.load_dotenv()

    print("Retrieving public IP address for VM instance...")
    client = compute_v1.InstancesClient()
    instance = client.get(project=os.getenv("GCP_PROJECT_ID"), instance=instance_name)

    if instance.network_interfaces and instance.network_interfaces[0].access_configs:
        return instance.network_interfaces[0].access_configs[0].nat_ip
    return None