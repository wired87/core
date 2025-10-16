import sys
import os
from typing import Union, List, Any

from artifact_registry.artifact_admin import ArtifactAdmin
from utils.run_subprocess import exec_cmd


class CloudBatchMaster:
    """
    Placeholder class definition, assuming the method resides here.
    """

    def __init__(self, project_id="aixr-401704", zone="us-central1-f", image_uri="qfs"):
        self.project_id = project_id
        self.zone = zone
        self.image_uri = image_uri

    def gcloud_create_instance(
            self,
            instance_name: str,
            # --- General Compute / Location ---
            project: str = None,
            zone: str = None,
            machine_type: str = "e2-standard-2",
            maintenance_policy: str = None,  # e.g., "TERMINATE" or "MIGRATE"
            provisioning_model: str = None,  # e.g., "STANDARD" or "SPOT"

            # --- Service Identity & Access ---
            service_account: str = None,
            scopes: Union[List[str], str] = None,  # List of scopes or comma-separated string

            # --- GPU Configuration ---
            gpu_type: str = None,  # e.g., "nvidia-tesla-t4"
            gpu_count: int = 0,

            # --- Boot Disk Configuration ---
            image: str = None,  # OS image URI (e.g., projects/cos-cloud/global/images/...)
            boot_disk_size: str = None,  # e.g., "10GB"
            boot_disk_type: str = None,  # e.g., "pd-balanced"
            boot_disk_device_name: str = None,

            # --- Container Configuration ---
            container_image: str = None,
            container_restart_policy: str = None,  # e.g., "always", "on-failure"
            env: dict = None,

            # --- Networking ---
            network: str = None,
            subnet: str = None,
            network_tier: str = None,  # e.g., "PREMIUM"
            stack_type: str = None,  # e.g., "IPV4_ONLY"
            tags: Union[List[str], str] = None,

            # --- Shielded VM ---
            shielded_secure_boot: bool = None,
            shielded_vtpm: bool = None,
            shielded_integrity_monitoring: bool = None,

            # --- Metadata/Labels ---
            labels: dict = None,

            **kwargs: Any
    ) -> List[str]:
        """
        Builds a gcloud compute instances create-with-container command
        as a list, including all specified optional parameters.
        """

        container_image = container_image or self.image_uri
        zone = zone or self.zone
        project = project or self.project_id

        cmd = ["gcloud", "compute", "instances", "create-with-container"]

        # --- Mandatory/Core Flags ---
        if instance_name:
            cmd.append(f"--name={instance_name}")
        if project:
            cmd.append(f"--project={project}")
        if zone:
            cmd.append(f"--zone={zone}")
        if machine_type:
            cmd.append(f"--machine-type={machine_type}")

        # --- Policy Flags ---
        if maintenance_policy:
            cmd.append(f"--maintenance-policy={maintenance_policy}")
        if provisioning_model:
            cmd.append(f"--provisioning-model={provisioning_model}")

        # --- Service Account & Scopes ---
        if service_account:
            cmd.append(f"--service-account={service_account}")
        if scopes:
            # Handle both list and string input for scopes
            scope_list = scopes if isinstance(scopes, (list, tuple)) else scopes.split(',')
            cmd.append(f"--scopes={','.join(scope_list)}")

        # --- Accelerator (GPU) Configuration ---
        if gpu_type and gpu_count > 0:
            accelerator_flag = f"count={gpu_count},type={gpu_type}"
            cmd.append(f"--accelerator={accelerator_flag}")
            # Mandate driver installation for ease of use
            cmd.append("--metadata=install-nvidia-driver=True")

        # --- Boot Disk Configuration ---
        if image:
            cmd.append(f"--image={image}")
        if boot_disk_size:
            cmd.append(f"--boot-disk-size={boot_disk_size}")
        if boot_disk_type:
            cmd.append(f"--boot-disk-type={boot_disk_type}")
        if boot_disk_device_name:
            cmd.append(f"--boot-disk-device-name={boot_disk_device_name}")

        # --- Container Configuration ---
        if container_image:
            cmd.append(f"--container-image={container_image}")
        if container_restart_policy:
            cmd.append(f"--container-restart-policy={container_restart_policy}")
        if env:
            env_str = ",".join(f"{k}={v}" for k, v in env.items())
            cmd.append(f"--container-env={env_str}")

        # --- Network Configuration ---
        if network or subnet or network_tier or stack_type:
            network_interface_parts = []

            if network_tier:
                network_interface_parts.append(f"network-tier={network_tier}")
            if stack_type:
                network_interface_parts.append(f"stack-type={stack_type}")
            if subnet:
                network_interface_parts.append(f"subnet={subnet}")
            elif network:
                # Fallback to network if subnet isn't specified
                network_interface_parts.append(f"network={network}")

            if network_interface_parts:
                cmd.append(f"--network-interface={','.join(network_interface_parts)}")

        if tags:
            tag_list = tags if isinstance(tags, (list, tuple)) else tags.split(',')
            cmd.append(f"--tags={','.join(tag_list)}")

        # --- Shielded VM Flags ---
        if shielded_secure_boot is True:
            cmd.append("--shielded-secure-boot")
        elif shielded_secure_boot is False:
            cmd.append("--no-shielded-secure-boot")

        if shielded_vtpm is True:
            cmd.append("--shielded-vtpm")
        if shielded_integrity_monitoring is True:
            cmd.append("--shielded-integrity-monitoring")

        # --- Labels ---
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in labels.items())
            cmd.append(f"--labels={label_str}")

        # --- Catch-all for other kwargs ---
        for key, value in kwargs.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is False:
                cmd.append(f"--no-{key}")
            elif value is not None:
                cmd.append(f"--{key}={value}")

        return cmd



class GcpDockerVmDeployer:
    """
    A class to deploy a local Docker image to a new GCP Compute Engine VM.
    All gcloud/docker commands are executed via a simple function.
    """
    def __init__(self):
        # --- Load and Assign Variables using os.getenv ---
        self.project_id = self._get_env("GCP_PROJECT_ID", "Project ID")
        self.region = self._get_env("GCP_REGION", "GCP Region")
        self.zone = self._get_env("GCP_ZONE", "Compute Engine Zone")

        self.port = 8000
        self.artifact_admin = ArtifactAdmin()



    def _get_env(self, key, description):
        """Helper function to retrieve env var or exit if missing."""
        value = os.getenv(key)
        if not value:
            print(f"❌ ERROR: Missing required environment variable: {key} ({description})")
            sys.exit(1)
        return value


    def create_vm(
            self,
            testing,
            instance_name,
            gpu_count,
            env,
            machine_type=None,
            gpu_type=None
    ):
        try:
            self.image_uri = self.artifact_admin.get_latest_image()
            if testing is True:
                self.test_instance(env)
            else:
                self.create_gpu_vm(
                    instance_name,
                    env,
                    gpu_count,
                    machine_type,
                    gpu_type
                )
            print("Created vm")
        except Exception as e:
            print(f"Err creating vm: {e}")

    def create_gpu_vm(
            self,
            instance_name,
            env,
            gpu_count=1,
            machine_type="nvidia-tesla-t4",
            gpu_type="nvidia-tesla-t4"
    ):
        gcloud_command = [
            "gcloud", "compute", "instances", "create-with-container",
            instance_name,
            f"--project={self.project_id}",
            f"--zone={self.zone}",
            f"--machine-type={machine_type}",
            "--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
            "--maintenance-policy=TERMINATE",
            "--provisioning-model=STANDARD",
            "--service-account=1004568990634-compute@developer.gserviceaccount.com",
            "--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append",
            f"--accelerator=count={gpu_count},type={gpu_type}",
            "--image=projects/cos-cloud/global/images/cos-109-17800-570-50",
            "--boot-disk-size=25GB",
            "--boot-disk-type=pd-balanced",
            f"--boot-disk-device-name={instance_name}",
            f"--container-image={self.image_uri}",
            "--container-restart-policy=always",
            "--no-shielded-secure-boot",
            "--shielded-vtpm",
            "--shielded-integrity-monitoring",
            "--labels=goog-ec-src=vm_add-gcloud,container-vm=cos-109-17800-570-50"
        ]
        if env:
            env_str = ",".join(f"{k}={v}" for k, v in env.items())
            gcloud_command.append(f"--container-env={env_str}")
        exec_cmd(gcloud_command)


    def test_instance(self, env):
        gcloud_command = [
            "gcloud", "compute", "instances", "create-with-container",
            "instance-20251016-143139",
            "--project=aixr-401704",
            "--zone=us-central1-f",  # Updated Zone
            "--machine-type=e2-medium",  # Updated Machine Type (CPU/Memory)
            "--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
            "--maintenance-policy=MIGRATE",  # Updated Maintenance Policy
            "--provisioning-model=STANDARD",
            "--service-account=1004568990634-compute@developer.gserviceaccount.com",
            "--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append",
            "--image=projects/cos-cloud/global/images/cos-109-17800-570-50",
            "--boot-disk-size=10GB",
            "--boot-disk-type=pd-balanced",
            "--boot-disk-device-name=instance-20251016-143139",
            "--container-image=qfs",
            "--container-restart-policy=always",
            "--no-shielded-secure-boot",
            "--shielded-vtpm",
            "--shielded-integrity-monitoring",
            "--labels=goog-ec-src=vm_add-gcloud,container-vm=cos-109-17800-570-50"
        ]
        if env:
            env_str = ",".join(f"{k}={v}" for k, v in env.items())
            gcloud_command.append(f"--container-env={env_str}")
        exec_cmd(gcloud_command)



    def gcloud_create_instance(
            self,
            instance_name,
            container_image=None,
            machine_type="nvidia-tesla-t4",
            zone=None,
            network=None,
            subnet=None,
            tags:list[str]=None,
            env:dict=None,
            gpu_count:int=1,
            gpu_type:str="nvidia-tesla-t4",
            **kwargs
    ):
        """
        Build a gcloud create-with-container command as a list.
        """

        container_image = container_image or self.image_uri
        zone = zone or self.zone

        cmd = ["gcloud", "compute", "instances", "create-with-container"]

        if instance_name:
            cmd.append(f"--name={instance_name}")
        if container_image:
            cmd.append(f"--container-image={container_image}")
        if env:
            env_str = ",".join(f"{k}={v}" for k, v in env.items())
            cmd.append(f"--container-env={env_str}")
        if machine_type:
            cmd.append(f"--machine-type={machine_type}")
        if zone:
            cmd.append(f"--zone={zone}")
        if network:
            cmd.append(f"--network={network}")
        if subnet:
            cmd.append(f"--subnet={subnet}")
        if tags:
            cmd.append(f"--tags={','.join(tags) if isinstance(tags, (list, tuple)) else tags}")

        # ----------------------------------------------------------------
        # GPU CONFIGURATION (NEW LOGIC)
        # ----------------------------------------------------------------
        if gpu_type and gpu_count > 0:
            # Format: --accelerator="type=GPU_TYPE,count=GPU_COUNT"
            accelerator_flag = f"type={gpu_type},count={gpu_count}"
            cmd.append(f"--accelerator={accelerator_flag}")

            # NOTE: You must also add the flag to install the GPU driver
            cmd.append("--metadata=install-nvidia-driver=True")




        for key, value in kwargs.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is False:
                cmd.append(f"--no-{key}")
            elif value is not None:
                cmd.append(f"--{key}={value}")

        return cmd

    def cleanup(self):
        """
        Optional cleanup function to delete the VM instance and the repository.
        """
        # Delete VM Instance
        vm_delete_command = f'gcloud compute instances delete {self.INSTANCE_NAME} --zone={self.ZONE} --quiet'
        print(f"🔄 Deleting VM: {self.INSTANCE_NAME}")
        exec_cmd(vm_delete_command)

        # Delete Artifact Registry Repository
        repo_delete_command = (
            f'gcloud artifacts repositories delete {self.REPO_NAME} '
            f'--location={self.region} --quiet'
        )
        print(f"🔄 Deleting Repository: {self.REPO_NAME}")
        exec_cmd(repo_delete_command)

        print("✅ Cleanup finished.")

