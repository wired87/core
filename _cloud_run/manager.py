"""
Cloud Run deployment manager.
Deploys container images to Google Cloud Run with configurable resources and env vars.
"""

import os
from typing import Dict, Optional

from google.cloud import run_v2
from google.protobuf import duration_pb2
from google.api_core.exceptions import NotFound, AlreadyExists
from google.auth.exceptions import DefaultCredentialsError

try:
    from auth.load_sa_creds import load_service_account_credentials
except ImportError:
    load_service_account_credentials = None


class CloudRunManager:
    """
    Deploy container images to Google Cloud Run programmatically.
    Supports custom resources (CPU, memory), env vars, and project credentials.
    """

    def __init__(
        self,
        project_id: str = None,
        region: str = None,
        credentials=None,
        **kwargs,
    ):
        """
        Args:
            project_id: GCP project ID (default: GCP_PROJECT_ID env)
            region: GCP region (default: GCP_REGION env)
            credentials: Optional credentials; else uses load_service_account_credentials()
        """
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")
        self.region = region or os.environ.get("GCP_REGION", "us-central1")
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

        creds = credentials or (load_service_account_credentials() if load_service_account_credentials else None)
        self.client = run_v2.ServicesClient(credentials=creds)

    def deploy(
        self,
        image_uri: str,
        service_name: str,
        env_vars: Dict[str, str] = None,
        cpu: int = 1,
        memory: str = "512Mi",
        min_instances: int = 0,
        max_instances: int = 10,
        timeout_seconds: int = 300,
        cpu_boost: bool = False,
        port: int = 8080,
        **kwargs,
    ) -> str:
        """
        Deploy or update a Cloud Run service.

        Args:
            image_uri: Full container image URI (e.g. us-central1-docker.pkg.dev/PROJECT/repo/image:tag)
            service_name: Cloud Run service name (DNS_LABEL)
            env_vars: Dict of env vars, e.g. {"KEY": "value"}
            cpu: vCPU count (1, 2, 4, 6, 8 or 0.08–1 for fractional).
            memory: Memory string, e.g. "256Mi", "512Mi", "1Gi", "2Gi"
            min_instances: Min instances (0 = scale to zero)
            max_instances: Max instances
            timeout_seconds: Request timeout
            cpu_boost: Enable CPU boost on startup
            port: Container port (Cloud Run sets PORT env automatically; use for container config)

        Returns:
            Service URL (e.g. https://service-xxx.run.app)
        """
        env_vars = env_vars or {}
        # Cloud Run v2: cpu as string "1", "2", etc.; memory as "256Mi", "1Gi", etc.
        limits = {"cpu": str(cpu), "memory": memory}

        env_list = [
            run_v2.EnvVar(name=k, value=str(v))
            for k, v in env_vars.items()
        ]

        container = run_v2.Container(
            image=image_uri,
            env=env_list,
            resources=run_v2.ResourceRequirements(
                limits=limits,
                cpu_boost=cpu_boost,
            ),
            ports=[run_v2.ContainerPort(container_port=port)] if port and port != 8080 else None,
        )

        template = run_v2.RevisionTemplate(
            containers=[container],
            scaling=run_v2.RevisionScaling(
                min_instance_count=min_instances,
                max_instance_count=max_instances,
            ),
            timeout=duration_pb2.Duration(seconds=timeout_seconds),
        )

        service = run_v2.Service(
            name=f"{self.parent}/services/{service_name}",
            template=template,
        )

        try:
            operation = self.client.create_service(
                parent=self.parent,
                service=service,
                service_id=service_name,
            )
            print(f"Deploying '{service_name}'...")
            response = operation.result()
            print(f"Deployed. URL: {response.uri}")
            return response.uri
        except AlreadyExists:
            print(f"Service '{service_name}' exists. Updating...")
            operation = self.client.update_service(service=service)
            response = operation.result()
            print(f"Updated. URL: {response.uri}")
            return response.uri
        except DefaultCredentialsError as e:
            raise RuntimeError(
                "Cloud Run auth failed. Set GOOGLE_APPLICATION_CREDENTIALS or run "
                "gcloud auth application-default login."
            ) from e

    def get_service(self, service_name: str) -> Optional[run_v2.Service]:
        """Get service by name. Returns None if not found."""
        try:
            return self.client.get_service(name=f"{self.parent}/services/{service_name}")
        except NotFound:
            return None

    def get_url(self, service_name: str) -> str:
        """Get public URL of the service. Returns empty string if not found."""
        svc = self.get_service(service_name)
        return svc.uri if svc else ""

    def delete_service(self, service_name: str) -> bool:
        """Delete the service. Returns True if deleted, False if not found."""
        try:
            operation = self.client.delete_service(name=f"{self.parent}/services/{service_name}")
            operation.result()
            print(f"Deleted service '{service_name}'")
            return True
        except NotFound:
            print(f"Service '{service_name}' not found")
            return False

    def list_services(self):
        """List all services in the region."""
        return self.client.list_services(parent=self.parent)

    @staticmethod
    def image_uri_artifact_registry(
        project_id: str,
        region: str,
        repo: str,
        image_name: str,
        tag: str = "latest",
    ) -> str:
        """Build Artifact Registry image URI."""
        return f"{region}-docker.pkg.dev/{project_id}/{repo}/{image_name}:{tag}"

    @staticmethod
    def image_uri_gcr(project_id: str, image_name: str, tag: str = "latest") -> str:
        """Build GCR image URI."""
        return f"gcr.io/{project_id}/{image_name}:{tag}"


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    project = os.environ.get("GCP_PROJECT_ID", "your-project")
    region = os.environ.get("GCP_REGION", "us-central1")

    mgr = CloudRunManager(project_id=project, region=region)

    # Example: deploy from Artifact Registry
    image = mgr.image_uri_artifact_registry(project, region, "qfs-repo", "qfs", "latest")
    url = mgr.deploy(
        image_uri=image,
        service_name="qfs-sim",
        env_vars={"ENV_ID": "test", "START_MODE": "SIM"},
        cpu=2,
        memory="2Gi",
        min_instances=0,
    )
    print(f"Service URL: {url}")
