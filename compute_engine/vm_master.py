import os
from google.cloud import compute_v1
from typing import List, Dict, Optional

from auth.load_sa_creds import load_service_account_credentials
from auth.set_gcp_auth_creds_path import set_gcp_auth_path

import dotenv

dotenv.load_dotenv()


class VMMaster:
    """
    A class to manage GCP Compute Engine VM instances.
    """

    def __init__(self, project_id: str = None, zone: str = "us-central1-a"):
        """
        Initializes the VMMaster class.

        Args:
            project_id (str, optional): The GCP project ID. Defaults to the value of the
                                        GCP_PROJECT_ID environment variable.
            zone (str, optional): The GCP zone. Defaults to "us-central1-a".
        """
        self.client = compute_v1.InstancesClient(
            credentials=load_service_account_credentials()
        )
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        self.zone = zone


    def create_instance(
        self,
        instance_name: str,
        machine_type: str = "e2-medium",
        source_image: str = "projects/debian-cloud/global/images/family/debian-11",
        network: str = "global/networks/default",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        service_account: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        gpu_type: Optional[str] = None,
        gpu_count: int = 0,
        container_image: Optional[str] = None,
        container_env: Optional[Dict[str, str]] = None,
        boot_disk_size_gb: int = 10,
        boot_disk_type: str = "pd-standard",
    ):
        """
        Creates a new, fully configurable VM instance.

        Args:
            instance_name (str): The name of the new VM instance.
            machine_type (str): The machine type of the new VM instance.
            source_image (str): The source image for the boot disk.
            network (str): The network to attach the instance to.
            tags (list): A list of network tags to apply to the instance.
            metadata (dict): A dictionary of metadata to apply to the instance.
            service_account (str): The service account to run the instance as.
            scopes (list): A list of API scopes to grant the instance.
            gpu_type (str): The type of GPU to attach to the instance.
            gpu_count (int): The number of GPUs to attach.
            container_image (str): The Docker image to run on the instance.
            container_env (dict): Environment variables for the container.
            boot_disk_size_gb (int): The size of the boot disk in GB.
            boot_disk_type (str): The type of the boot disk.
        """
        print(f"Creating new VM instance: {instance_name}")
        try:
            instance_config = {
                "name": instance_name,
                "machine_type": f"zones/{self.zone}/machineTypes/{machine_type}",
                "disks": [
                    {
                        "boot": True,
                        "auto_delete": True,
                        "initialize_params": {
                            "source_image": source_image,
                            "disk_size_gb": boot_disk_size_gb,
                            "disk_type": f"zones/{self.zone}/diskTypes/{boot_disk_type}",
                        },
                    }
                ],
                "network_interfaces": [
                    {
                        "network": network,
                        "access_configs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
                    }
                ],
                "metadata": {"items": []},
            }

            if tags:
                instance_config["tags"] = {"items": tags}

            if metadata:
                for key, value in metadata.items():
                    instance_config["metadata"]["items"].append({"key": key, "value": value})

            if service_account:
                instance_config["service_accounts"] = [
                    {"email": service_account, "scopes": scopes or ["https://www.googleapis.com/auth/cloud-platform"]}
                ]

            if gpu_type and gpu_count > 0:
                instance_config["guest_accelerators"] = [
                    {
                        "accelerator_count": gpu_count,
                        "accelerator_type": f"zones/{self.zone}/acceleratorTypes/{gpu_type}",
                    }
                ]
                instance_config["scheduling"] = {"on_host_maintenance": "TERMINATE"}

            if container_image:
                spec = f"spec:\n  containers:\n    - name: {instance_name}\n      image: {container_image}\n"
                if container_env:
                    spec += "      env:\n"
                    for key, value in container_env.items():
                        spec += f"        - name: {key}\n          value: {value}\n"
                instance_config["metadata"]["items"].append({"key": "gce-container-declaration", "value": spec})

            operation = self.client.insert(project=self.project_id, zone=self.zone, instance_resource=instance_config)
            operation.result()
            print(f"VM instance {instance_name} created successfully.")
        except Exception as e:
            print("Err create_instance", e)

    def delete_instance(self, instance_name: str):
        """
        Deletes a VM instance.

        Args:
            instance_name (str): The name of the VM instance to delete.
        """
        print(f"Deleting VM instance: {instance_name}")
        operation = self.client.delete(project=self.project_id, zone=self.zone, instance=instance_name)
        operation.result()
        print(f"VM instance {instance_name} deleted successfully.")

    def start_instance(self, instance_name: str):
        """
        Starts a VM instance.

        Args:
            instance_name (str): The name of the VM instance to start.
        """
        print(f"Starting VM instance: {instance_name}")
        operation = self.client.start(project=self.project_id, zone=self.zone, instance=instance_name)
        operation.result()
        print(f"VM instance {instance_name} started successfully.")

    def stop_instance(self, instance_name: str):
        """
        Stops a VM instance.

        Args:
            instance_name (str): The name of the VM instance to stop.
        """
        print(f"Stopping VM instance: {instance_name}")
        operation = self.client.stop(project=self.project_id, zone=self.zone, instance=instance_name)
        operation.result()
        print(f"VM instance {instance_name} stopped successfully.")

    def get_public_ip(self, instance_name: str) -> str:
        """
        Gets the public IP address of a VM instance.

        Args:
            instance_name (str): The name of the VM instance.

        Returns:
            str: The public IP address of the VM instance, or None if not found.
        """
        print(f"Retrieving public IP address for VM instance: {instance_name}")
        instance = self.client.get(project=self.project_id, zone=self.zone, instance=instance_name)
        if instance.network_interfaces and instance.network_interfaces[0].access_configs:
            return instance.network_interfaces[0].access_configs[0].nat_ip
        return None

    def list_instances(self) -> list:
        """
        Lists all VM instances in the project.

        Returns:
            list: A list of all VM instance names.
        """
        print("Listing all VM instances...")
        names = []
        request = compute_v1.AggregatedListInstancesRequest(project=self.project_id)
        for _, response in self.client.aggregated_list(request=request):
            if response.instances:
                for instance in response.instances:
                    names.append(instance.name)
        return names

    def execute_startup_script(self, instance_name: str, script: str):
        """
        Executes a startup script on a VM instance.

        Args:
            instance_name (str): The name of the VM instance.
            script (str): The startup script to execute.
        """
        print(f"Executing startup script on VM instance: {instance_name}")
        instance = self.client.get(project=self.project_id, zone=self.zone, instance=instance_name)
        metadata = instance.metadata
        metadata.items.append({"key": "startup-script", "value": script})
        operation = self.client.set_metadata(project=self.project_id, zone=self.zone, instance=instance_name, metadata_resource=metadata)
        operation.result()
        print(f"Startup script executed on VM instance {instance_name}.")


    def main(
            self,
            test_instance_name= "test-instance-from-vmmaster",
            vcpu_count: int = 8,
            memory_mb: int = 32768,
            gpu_type: str = "nvidia-tesla-t4",
            gpu_count: int = 1,
            container_image="tensorflow/tensorflow:latest-gpu",
    ):
        # Construct the custom machine type string
        machine_type = f"n1-custom-{vcpu_count}-{memory_mb}"

        instance_config = {
            "instance_name": test_instance_name,
            "machine_type": machine_type,
            "source_image": "projects/cos-cloud/global/images/family/cos-stable",
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "container_image": container_image,
            "container_env": {"PORT": "8080"},
            "tags": ["test-vm"],
        }

        try:
            # Create the VM
            self.create_instance(**instance_config)

            # Get and print the public IP
            public_ip = self.get_public_ip(test_instance_name)
            print(f"Public IP address of {test_instance_name}: {public_ip}")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Delete the VM
            try:
                input(f"Press Enter to delete the VM instance '{test_instance_name}'...")
                self.delete_instance(test_instance_name)
            except Exception as e:
                print(f"An error occurred during cleanup: {e}")


if __name__ == "__main__":
    # Initialize the VMMaster
    set_gcp_auth_path()
    vm_master = VMMaster()

    # Example of calling main with custom CPU and GPU settings
    vm_master.main(
        vcpu_count=4,
        memory_mb=16384,  # 16 GB
        gpu_type="nvidia-tesla-t4",
        gpu_count=1
    )