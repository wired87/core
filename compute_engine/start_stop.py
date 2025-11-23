from google.cloud import compute_v1

import os

GCP_ID = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def start_vm_instance(instance_name):
    print("Starting VM instance...")
    client = compute_v1.InstancesClient()
    operation = client.start(project=GCP_ID, instance=instance_name)
    operation.result()

def stop_vm_instance(instance_name):
    print("Stopping VM instance...")
    client = compute_v1.InstancesClient()
    operation = client.stop(project=GCP_ID, instance=instance_name)
    operation.result()