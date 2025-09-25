import time
from google.cloud import run_v2
from google.api_core.exceptions import NotFound, Conflict
from google.auth.exceptions import DefaultCredentialsError


class CloudRunMaster:
    """
    A class to manage the full lifecycle of a Google Cloud Run service.

    This class handles the deployment, management, and deletion of a Cloud Run
    service using the v2 API.
    """

    def __init__(self, project_id: str, region: str):
        """
        Initializes the CloudRunMaster with project and region information.

        Args:
            project_id (str): The Google Cloud project ID.
            region (str): The Google Cloud region where the service will be deployed.
        """
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{self.project_id}/locations/{self.region}"
        self.client = run_v2.ServicesClient()

    def deploy_service(self, service_name: str, image_uri: str, resources: dict = None, cpu_boost: bool = False,
                       timeout_seconds: int = 300) -> str:
        """
        Deploys or updates a Cloud Run service.

        Args:
            service_name (str): The name of the Cloud Run service.
            image_uri (str): The container image URI (e.g., gcr.io/my-project/my-image:tag).
            resources (dict, optional): A dictionary specifying CPU and memory limits.
                                        Example: {"cpu_idle": True, "cpu": 1000, "memory": "256Mi"}
                                        CPU is in millicores (1000 = 1 CPU), memory in MiB.
            cpu_boost (bool): Enables or disables CPU boost on startup. Defaults to False.
            timeout_seconds (int): The request timeout in seconds.

        Returns:
            str: The public URL of the deployed service.
        """
        print(f"Deploying service '{service_name}' with image '{image_uri}'...")
        try:
            # Construct the service deployment request
            service_request = run_v2.Service(
                name=f"{self.parent}/services/{service_name}",
                template=run_v2.RevisionTemplate(
                    containers=[
                        run_v2.Container(
                            image=image_uri,
                            resources=run_v2.ResourceRequirements(
                                cpu_boost=cpu_boost,
                                limits=resources
                            ) if resources else None
                        )
                    ],
                    scaling=run_v2.K8sV1Scaling(
                        min_instance_count=0
                    ),
                    timeout_seconds=timeout_seconds,
                )
            )

            operation = self.client.create_service(
                parent=self.parent,
                service=service_request,
                service_id=service_name,
            )
            print(f"Deployment started. Waiting for operation to complete...")
            response = operation.result()
            print("Deployment successful.")
            return response.uri

        except Conflict:
            print(f"Service '{service_name}' already exists. Updating it...")
            # If the service exists, update it instead of creating a new one
            service_request = run_v2.Service(
                name=f"{self.parent}/services/{service_name}",
                template=run_v2.RevisionTemplate(
                    containers=[
                        run_v2.Container(
                            image=image_uri,
                            resources=run_v2.ResourceRequirements(
                                cpu_boost=cpu_boost,
                                limits=resources
                            ) if resources else None
                        )
                    ],
                    scaling=run_v2.K8sV1Scaling(
                        min_instance_count=0
                    ),
                    timeout_seconds=timeout_seconds,
                )
            )
            operation = self.client.update_service(service=service_request)
            print(f"Update started. Waiting for operation to complete...")
            response = operation.result()
            print("Update successful.")
            return response.uri
        except DefaultCredentialsError as e:
            raise RuntimeError(
                "Could not authenticate. Make sure you have authenticated "
                "via `gcloud auth application-default login` and have the "
                "required IAM permissions."
            ) from e

    def get_public_domain(self, service_name: str) -> str:
        """
        Gets the public domain (URL) of a Cloud Run service.

        Args:
            service_name (str): The name of the Cloud Run service.

        Returns:
            str: The public URL of the service.
        """
        print(f"Fetching public domain for service '{service_name}'...")
        try:
            service = self.client.get_service(name=f"{self.parent}/services/{service_name}")
            return service.uri
        except NotFound:
            print(f"Service '{service_name}' not found.")
            return ""

    def delete_service(self, service_name: str):
        """
        Deletes a Cloud Run service by name.

        Args:
            service_name (str): The name of the Cloud Run service to delete.
        """
        print(f"Deleting service '{service_name}'...")
        try:
            operation = self.client.delete_service(name=f"{self.parent}/services/{service_name}")
            print(f"Deletion started. Waiting for operation to complete...")
            operation.result()
            print(f"Service '{service_name}' deleted successfully.")
        except NotFound:
            print(f"Service '{service_name}' not found. No action taken.")

    @staticmethod
    def get_image_uri_from_gcr(project_id: str, image_name: str, tag: str = "latest"):
        """
        Helper method to construct a GCR image URI.

        Args:
            project_id (str): Your Google Cloud project ID.
            image_name (str): The name of your container image.
            tag (str): The image tag. Defaults to "latest".

        Returns:
            str: The full image URI.
        """
        return f"gcr.io/{project_id}/{image_name}:{tag}"


# Example usage
if __name__ == "__main__":
    # --- IMPORTANT: Configure these variables with your own values ---
    PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your project ID
    REGION = "us-central1"  # Replace with your desired region
    SERVICE_NAME = "my-test-service"
    IMAGE_NAME = "hello-world"
    IMAGE_TAG = "latest"
    # --- End of configuration ---

    if PROJECT_ID == "YOUR_PROJECT_ID":
        print("Please configure the PROJECT_ID variable before running this script.")
    else:
        # Create an instance of the master class
        master = CloudRunMaster(project_id=PROJECT_ID, region=REGION)
        image_uri = master.get_image_uri_from_gcr(PROJECT_ID, IMAGE_NAME, IMAGE_TAG)

        # 1. Deploy the service
        service_url = master.deploy_service(
            service_name=SERVICE_NAME,
            image_uri=image_uri,
            resources={"cpu": 1000, "memory": "512Mi"}
        )
        if service_url:
            print(f"Service deployed. Public URL: {service_url}")

            # Wait a moment to ensure the service is fully ready for a separate request
            time.sleep(10)

            # 2. Get the public domain (URL)
            retrieved_url = master.get_public_domain(SERVICE_NAME)
            print(f"Retrieved public domain: {retrieved_url}")

            # 3. Clean up and delete the service
            master.delete_service(SERVICE_NAME)
