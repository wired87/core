"""
Vertex AI Workbench manager.
Deploy custom container images from Artifact Registry to Workbench and execute there.
All parameters are explicit for maximum transparency.
"""

import os
from typing import Dict, Optional, List, Any

from google.cloud import aiplatform
from google.cloud.aiplatform import CustomJob

from auth.load_sa_creds import load_service_account_credentials
from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin


def _run(cmd: List[str]) -> Optional[str]:
    """Run gcloud command. Returns stdout or None on failure."""
    import subprocess
    try:
        # On Windows, use shell=True so gcloud.cmd is found (PATH from shell)
        use_shell = os.name == "nt"
        args = " ".join(cmd) if use_shell else cmd
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,
            shell=use_shell,
        )
        if r.returncode != 0:
            print(f"[VertexWorkbench] Command failed: {r.stderr or r.stdout}")
            return None
        return r.stdout.strip() if r.stdout else ""
    except subprocess.TimeoutExpired as e:
        print(f"[VertexWorkbench] Timeout: {e}")
        return None
    except FileNotFoundError as e:
        print(f"[VertexWorkbench] gcloud not found. Is the Google Cloud SDK installed and on PATH? {e}")
        return None
    except Exception as e:
        print(f"[VertexWorkbench] Error: {e}")
        return None


class VertexTrainerManager:
    def __init__(
            self,
            *,
            project_id: str = None,
            location: str = None,  # Wir nutzen das als Region
            credentials=None,
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")

        raw_location = location or os.environ.get("GCP_REGION", "us-central1")

        # Falls doch mal eine Zone reurutscht, schneiden wir das Suffix ab (z.B. -b)
        self.location = "-".join(raw_location.split("-")[:2])

        self.api_endpoint = f"{self.location}-aiplatform.googleapis.com"
        self.credentials = credentials
        self.client = aiplatform.gapic.JobServiceClient(
            credentials=load_service_account_credentials(),
            client_options={
                "api_endpoint": self.api_endpoint,
            }
        )


    def create_custom_job(
            self,
            display_name: str,
            container_image_uri: str,
            container_envs: Dict[str, Any],
    ):
        # todo gu usr validation from usr object (use_gpu bool)
        custom_job:CustomJob = {
            "display_name": display_name,
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": "n1-standard-4",
                            #"accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4,
                            #"accelerator_count": None, # 1 hits quota
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": container_image_uri,
                            "command": [],
                            "args": [],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in container_envs.items()
                            ],
                        },
                    }
                ]
            },
        }
        parent = f"projects/{self.project_id}/locations/{self.location}"
        response = self.client.create_custom_job(
            parent=parent,
            custom_job=custom_job,
        )
        print("response:", response)


    def deploy(
        self,
        *,
        instance_name: str,
        container_repository: str,
        container_tag: str = "latest",
        machine_type: str = "n1-standard-4",
        boot_disk_size_gb: int = 100,
        boot_disk_type: str = "PD_STANDARD",
        data_disk_size_gb: int = 100,
        data_disk_type: str = "PD_STANDARD",
        accelerator_type: str = None,
        accelerator_core_count: int = 0,
        service_account_email: str = None,
        network: str = None,
        subnet: str = None,
        subnet_region: str = None,
        metadata: Dict[str, str] = None,
        labels: Dict[str, str] = None,
        post_startup_script: str = None,
        container_env_vars: Dict[str, str] = None,
        container_allow_fuse: bool = False,
        container_custom_params: str = None,
        disable_public_ip: bool = False,
        install_gpu_driver: bool = False,
        async_create: bool = False,
    ) -> Optional[str]:
        """
        Create a Vertex AI Workbench instance with a custom container.

        All parameters are explicit. Pass only what you need; rest use defaults.

        Args:
            instance_name: Unique name (1-63 chars, lowercase, numbers, dashes).
            container_repository: Full image path, e.g. us-central1-docker.pkg.dev/PROJECT/repo/image.
            container_tag: Image tag. Default: latest.
            machine_type: e.g. n1-standard-4, n1-highmem-4.
            boot_disk_size_gb: Boot disk size. Default: 100.
            boot_disk_type: PD_STANDARD, PD_SSD, PD_BALANCED, PD_EXTREME.
            data_disk_size_gb: Data disk size. Default: 100.
            data_disk_type: Same as boot_disk_type.
            accelerator_type: e.g. NVIDIA_TESLA_T4, NVIDIA_TESLA_A100.
            accelerator_core_count: GPU count when accelerator_type set.
            service_account_email: SA for the instance.
            network: VPC network (projects/PROJECT/global/networks/NAME).
            subnet: Subnet (projects/PROJECT/regions/REGION/subnetworks/NAME).
            subnet_region: Region of subnet if subnet given.
            metadata: Custom metadata key=value.
            labels: Instance labels.
            post_startup_script: gs://bucket/path/to/script.sh
            container_env_vars: Env vars for container (KEY=value).
            container_allow_fuse: Enable Cloud Storage FUSE.
            container_custom_params: Extra nerdctl params, e.g. --v /mnt/disk1:/mnt/disk1.
            disable_public_ip: No public IP.
            install_gpu_driver: Install GPU driver (when using GPU).
            async_create: Return immediately, don't wait for create.

        Returns:
            Proxy URL if successful and not async, else None.
        """
        cmd = [
            "gcloud", "workbench", "instances", "create", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
            f"--container-repository={container_repository}",
            f"--container-tag={container_tag}",
            f"--machine-type={machine_type}",
            f"--boot-disk-size={boot_disk_size_gb}",
            f"--boot-disk-type={boot_disk_type}",
            f"--data-disk-size={data_disk_size_gb}",
            f"--data-disk-type={data_disk_type}",
        ]

        if accelerator_type and accelerator_core_count > 0:
            cmd.extend([
                f"--accelerator-type={accelerator_type}",
                f"--accelerator-core-count={accelerator_core_count}",
            ])
            if install_gpu_driver:
                cmd.append("--install-gpu-driver")

        if service_account_email:
            cmd.append(f"--service-account-email={service_account_email}")
        if network:
            cmd.append(f"--network={network}")
        if subnet:
            cmd.append(f"--subnet={subnet}")
        if subnet_region:
            cmd.append(f"--subnet-region={subnet_region}")
        if disable_public_ip:
            cmd.append("--disable-public-ip")

        meta_parts = []
        if metadata:
            for k, v in metadata.items():
                meta_parts.append(f"{k}={v}")
        if post_startup_script:
            meta_parts.append(f"post-startup-script={post_startup_script}")
        if container_env_vars:
            env_str = ",".join(f"{k}={v}" for k, v in container_env_vars.items())
            meta_parts.append(f"container-env-file={env_str}")
        if container_allow_fuse:
            meta_parts.append("container-allow-fuse=true")
        if container_custom_params:
            meta_parts.append(f"container-custom-params={container_custom_params}")

        if meta_parts:
            cmd.append("--metadata=" + ",".join(meta_parts))

        if labels:
            lbl = ",".join(f"{k}={v}" for k, v in labels.items())
            cmd.append(f"--labels={lbl}")

        if async_create:
            cmd.append("--async")

        print(f"[VertexWorkbench] Creating instance '{instance_name}'...")
        out = _run(cmd)
        if out is None:
            print("[VertexWorkbench] Create failed.")
            return None

        if async_create:
            return "async"
        return self.get_proxy_url(instance_name=instance_name)

    def get_proxy_url(self, *, instance_name: str) -> Optional[str]:
        """Get proxy URL for the instance. Returns None if not found or not ready."""
        cmd = [
            "gcloud", "workbench", "instances", "describe", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
            "--format=value(proxyUri)",
        ]
        out = _run(cmd)
        return out.strip() if out else None

    def describe(self, *, instance_name: str) -> Optional[str]:
        """Describe instance. Returns full output or None."""
        cmd = [
            "gcloud", "workbench", "instances", "describe", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
        ]
        return _run(cmd)

    def start(self, *, instance_name: str, wait: bool = True) -> bool:
        """Start the instance. Returns True on success."""
        cmd = [
            "gcloud", "workbench", "instances", "start", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
        ]
        if not wait:
            cmd.append("--async")
        out = _run(cmd)
        return out is not None

    def stop(self, *, instance_name: str, wait: bool = True) -> bool:
        """Stop the instance. Returns True on success."""
        cmd = [
            "gcloud", "workbench", "instances", "stop", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
        ]
        if not wait:
            cmd.append("--async")
        out = _run(cmd)
        return out is not None

    def delete(self, *, instance_name: str, wait: bool = True) -> bool:
        """Delete the instance. Returns True on success."""
        cmd = [
            "gcloud", "workbench", "instances", "delete", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
            "--quiet",
        ]
        if not wait:
            cmd.append("--async")
        out = _run(cmd)
        return out is not None

    def execute(
        self,
        *,
        instance_name: str,
        command: str,
        user: str = "jupyter",
        zone: str = None,
    ) -> Optional[str]:
        """
        Execute a command on the Workbench instance via gcloud compute ssh.

        Args:
            instance_name: Instance name (VM name may differ; use instance name).
            command: Shell command to run.
            user: SSH user (default jupyter for Workbench).
            zone: Override zone for SSH target.

        Returns:
            Command output or None.
        """
        zone = zone or self.location
        # Workbench VM name is typically the instance name
        vm_name = instance_name
        cmd = [
            "gcloud", "compute", "ssh", f"{user}@{vm_name}",
            f"--project={self.project_id}",
            f"--zone={zone}",
            f"--command={command}",
        ]
        return _run(cmd)

    def update_container(
        self,
        *,
        instance_name: str,
        container_repository: str,
        container_tag: str = "latest",
    ) -> bool:
        """Update the container image on an existing instance."""
        cmd = [
            "gcloud", "workbench", "instances", "update", instance_name,
            f"--project={self.project_id}",
            f"--location={self.location}",
            f"--container-repository={container_repository}",
            f"--container-tag={container_tag}",
        ]
        out = _run(cmd)
        return out is not None



if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    project = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_ZONE") or os.environ.get("GCP_REGION", "us-central1-b")

    mgr = VertexTrainerManager(project_id=project, location=location)
    registry = ArtifactAdmin()

    url = mgr.create_custom_job(
        display_name="qfs-workbench-01",
        container_image_uri=registry.get_latest_image(),
        container_envs={}
    )
    if url:
        print(f"Workbench URL: {url}")
