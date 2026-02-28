import sys
import os
from typing import Union, List, Any

from qbrain.auth.set_gcp_auth_creds_path import set_gcp_auth_path
from compute_engine import VMMaster
from create_env import EnvCreatorProcess
from qbrain.utils.run_subprocess import exec_cmd


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
            for k, v in env.items():
                cmd.append(f"--container-env={k}={v}")

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



class DeploymentHandler(VMMaster):
    """
    A class to deploy a local Docker image to a new GCP Compute Engine VM.
    All gcloud/docker commands are executed via a simple function.
    """
    def __init__(self, user_id):
        # Load env first so VMMaster gets correct project/zone
        project_id = self._get_env("GCP_PROJECT_ID", "Project ID")
        zone = self._get_env("GCP_ZONE", "Compute Engine Zone")
        region = self._get_env("GCP_REGION", "GCP Region")
        VMMaster.__init__(self, project_id=project_id, zone=zone)
        self.project_id = project_id
        self.region = region
        self.zone = zone

        self.user_id = user_id
        self.env_creator = EnvCreatorProcess(self.user_id)
        self.port = 8000

        self.test_vm_cfg: dict[str, Any] = {
            "instance_name": "test-vm-minimal-01",
            "machine_type": "e2-micro",
            "source_image": "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11", #"projects/debian-cloud/global/images/family/debian-11",
            "network": "global/networks/default",
            "tags": ["test-vm", "ephemeral"],
            "metadata": {
                "owner": "benedikt_test",
                "env": "dev"
            },
            #"service_account": os.getenv("SACC_NAME"),
            "scopes": [
                "https://www.googleapis.com/auth/devstorage.read_only",
                "https://www.googleapis.com/auth/logging.write"
            ],
            "gpu_type": None,
            "gpu_count": 0,
            "container_image": "busybox:latest",
            "container_env": {
                "START_MODE": "TEST",
                "TIMEOUT": "300"
            },
            "boot_disk_size_gb": 10,
            "boot_disk_type": "pd-balanced",
        }

    def _get_env(self, key, description):
        """Helper function to retrieve env var or exit if missing."""
        value = os.getenv(key)
        if not value:
            print(f"❌ ERROR: Missing required environment variable: {key} ({description})")
            sys.exit(1)
        return value


    def get_prod_vm_cfg(
            self,
            instance_name: str,
            container_image: str,
            container_env: dict,
            testing: bool = False  # Neuer Parameter
    ) -> dict[str, Any]:
        print(f"Generiere VM-Config (Mode: {'TESTING' if testing else 'PROD'}) für {instance_name}")

        # Standard-Werte für Produktion
        machine_type = "n1-standard-16"
        gpu_type = "nvidia-tesla-t4"
        gpu_count = 1
        tags = [
            "allow-ssh-via-iap",
            "http-server",
            "https-server",
            "lb-health-check",
        ]
        env_label = "prod"
        # Fallback für Testing (2 Cores, No GPU)
        if testing is True:
            machine_type = "e2-standard-2"  # 2 Cores, günstigere E2-Serie
            gpu_type = None
            gpu_count = 0
            env_label = "test"

        startup_script_content = self.generate_startup_script(
            container_env,
            container_image,
        )

        config = {
            "instance_name": instance_name,
            "machine_type": machine_type,
            "source_image": "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11",#"projects/debian-cloud/global/images/family/debian-11",
            "network": "global/networks/default",
            "tags": tags,
            "metadata": {
                "owner": instance_name,
                "env": env_label,
                "project_id": "aixr-401704",
                "startup-script":startup_script_content
            },
            #
            "scopes": [
                "https://www.googleapis.com/auth/cloud-platform"
            ],
            "container_image": container_image,
            "container_env": container_env,
            "boot_disk_size_gb": 30,
            "boot_disk_type": "pd-balanced",
        }

        # Nur GPU-Felder hinzufügen, wenn wir nicht im Testmodus sind
        if not testing:
            config["gpu_type"] = gpu_type
            config["gpu_count"] = gpu_count
        return config


    def generate_startup_script2(
            self,
            container_env,
            image_name,
    ):
        """
        Erstellt ein dynamisches Bash-Script für den VM-Start.
        """
        # Wandelt das Dictionary {KEY: VAL} in Docker-Format "-e KEY=VAL" um
        env_flags = " ".join([f"-e {k}='{v}'" for k, v in container_env.items()])

        script = f"""
        #!/bin/bash
        
        
        apt-get update
        apt-get install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        
        
        
        docker run --restart always --gpus all \\
            {env_flags} \\
            {image_name}         
        echo "Startup script finished - Container is running."
        """
        return script

    import os

    def generate_startup_script(
            self,
            container_env: dict,
            container_image: str = "us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs:latest",
            image_name: str = None,
    ):
        """
        Erstellt ein Bash-Script, das Docker + GPU-Treiber installiert und den Container startet.
        Extrahiert Default-Werte dynamisch aus dem container_image Pfad.
        """

        # 1. Fallback für den lokalen Container-Namen (z.B. extrahiert 'qfs' aus dem Pfad)
        if image_name is None:
            # Extrahiert den Teil nach dem letzten '/' und vor dem ':'
            # us-central1-docker.pkg.dev/.../qfs:latest -> qfs
            image_name = container_image.split('/')[-1].split(':')[0]

        # 2. Wandelt das Dictionary in Docker-Format "-e KEY=VAL" um
        env_flags = " ".join([f"-e {k}='{v}'" for k, v in container_env.items()])

        # 3. Extrahiere die Region für die Registry-Auth (us-central1, europe-west10, etc.)
        try:
            registry_region = container_image.split('-docker.pkg.dev')[0].split('/')[-1]
        except IndexError:
            registry_region = "us-central1"  # Sicherer Fallback

        # WICHTIG: Text nach links gerückt, damit keine Leerzeichen vor #!/bin/bash landen
        script = f"""#!/bin/bash
        set -e
    
        echo "--- Starte System-Vorbereitung ---"
        
        # 1. Docker installieren (falls nicht vorhanden)
        if ! command -v docker &> /dev/null; then
            apt-get update
            apt-get install -y docker.io
            systemctl start docker
            systemctl enable docker
        fi
    
        # 2. NVIDIA Container Toolkit installieren
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update
            apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker
        fi
    
        # 3. Authentifizierung an der Google Artifact Registry
        gcloud auth configure-docker {registry_region}-docker.pkg.dev --quiet
    
        # 4. Container ziehen und starten
        echo "--- Starte Container: {image_name} ---"
        docker pull {container_image}
    
        # Falls ein alter Container mit gleichem Namen läuft, diesen entfernen
        docker rm -f {image_name} || true
    
        docker run -d \\
            --name {image_name} \\
            --restart always \\
            --gpus all \\
            {env_flags} \\
            {container_image}
    
        echo "--- Startup script finished - Container is running ---"
        """

        # Debug Output
        print(f"Startup Script erstellt für Image: {container_image}")
        print(f"Lokaler Container-Name: {image_name}")

        # Lokales Speichern zur Kontrolle
        save_path = r"C:\Users\bestb\PycharmProjects\BestBrain\test_startup.sh"
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", newline='\n') as outfile:
                outfile.write(script)
        except Exception as e:
            print(f"Hinweis: Konnte Datei nicht lokal speichern ({e}), gebe Script aber zurück.")

        return script








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
            for k, v in env.items():
                gcloud_command.append(f"--container-env={k}={v}")
        exec_cmd(gcloud_command)


    def test_instance(self, env):
        print("create test_instance:", env)
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
            "--labels=goog-ec-src=vm_add-gcloud,container-vm=cos-109-17800-570-50",
            "--container-image=memcached",
        ]
        if env:
            for k, v in env.items():
                gcloud_command.append(f"--container-env={k}={v}")
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
            for k, v in env.items():
                cmd.append(f"--container-env={k}={v}")
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

    def stop_vm(self, instance_name):
        print(f"Stopping VM: {instance_name}")
        cmd = ["gcloud", "compute", "instances", "delete", instance_name, f"--zone={self.zone}", "--quiet"]
        exec_cmd(cmd)

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


if __name__ == "__main__":
    set_gcp_auth_path()
    dhandler= DeploymentHandler("123")
    dhandler.create_vm(
        testing=True,
        instance_name="999"
    )