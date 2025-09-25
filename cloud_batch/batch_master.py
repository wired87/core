import os

import google.cloud.batch_v1 as batch
from google.api_core.exceptions import Conflict, NotFound
from typing import List, Dict

from artifact_registry.artifact_admin import ArtifactAdmin

from cloud_batch.batch_admin import CloudBatchAdmin

import dotenv

from fb_core.real_time_database import FirebaseRTDBManager

dotenv.load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")

class CloudBatchMaster:
    """
    A class to manage the full lifecycle of a Google Cloud Batch job.

    This class handles the creation, monitoring, and deletion of a batch job
    using the official Python client library.
    """

    def __init__(
            self,
            project_id=PROJECT_ID,
            region=REGION,
            db_manager=None
    ):
        """
        Initializes the CloudBatchMaster with project and region information.

        Args:
            project_id (str): The Google Cloud project ID.
            region (str): The Google Cloud region where the job will run.
        """
        self.project_id = project_id
        self.region = region or "europe-west10"
        self.parent = f"projects/{self.project_id}/locations/{self.region}"
        self.client = batch.BatchServiceClient()

        # classes
        self.db_manager = db_manager or FirebaseRTDBManager()

        self.admin = CloudBatchAdmin(
            self.project_id,
            self.region,
            self.client,
            db_manager
        )


    def create_batch_job(self,
                         job_name: str,
                         #command: List[str],
                         image_uri,
                         task_count: int,
                         cpu_milli: int,
                         memory_gib: int,
                         env_vars: Dict[str, str],
                         accelerators: List[Dict[str, str]] = None) -> str:
        """
        Creates and submits a new Cloud Batch job.

        Args:
            job_name (str): The unique name for the job.
            image_uri (str): The container image URI from Artifact Registry.
                             Example: us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest
            task_count (int): The number of tasks to run in parallel.
            cpu_milli (int): The number of CPU milli-cores per task (1000 = 1 CPU).
            memory_gib (int): The memory in GiB per task.
            env_vars (Dict[str, str]): A dictionary of environment variables to set.
            accelerators (List[Dict[str, str]], optional): A list of accelerators to attach.
                                                            Example: [{"type": "nvidia-tesla-t4", "count": 1}]

        Returns:
            str: The name of the created job resource.
        """
        print(f"Creating job '{job_name}'...")
        runnable = batch.Runnable()
        runnable.container = batch.Runnable.Container()

        runnable.container.image_uri = image_uri

        """
        runnable.container.entrypoint = command[0]
        runnable.container.commands = command[1:]
        """

        runnable.container.block_external_network=False

        # Set environment variables
        if env_vars:
            runnable.environment = batch.Environment()
            runnable.environment.variables = env_vars

        task = batch.TaskSpec(
            runnables=[runnable]
        )

        task.compute_resource = batch.ComputeResource(
            cpu_milli=cpu_milli,
            memory_mib=memory_gib * 1024
        )

        if accelerators:
            accelerator_list = []
            for acc in accelerators:
                accelerator = batch.Accelerator(
                    type_=acc["type"],
                    count=acc["count"]
                )
                accelerator_list.append(accelerator)
            task.compute_resource.accelerators = accelerator_list

        group = batch.TaskGroup(
            task_spec=task,
            task_count=task_count
        )

        job = batch.Job(
            task_groups=[group],#?
            name=job_name,
            #uid=job_name,
            #allocation_policy=batch.AllocationPolicy(location=batch.LocationPolicy(allowed_locations=[self.region])),
            labels={"job_type": "custom-batch-job"}
        )

        try:
            request = batch.CreateJobRequest(
                parent=self.parent,
                job_id=job_name,
                job=job,
            )
            response = self.client.create_job(request=request)
            print(f"Job '{job_name}' created successfully.")
            return response.name
        except Conflict:
            print(
                f"Job '{job_name}' already exists. Please delete the existing job before creating a new one with the same name.")
            return ""

    def get_job_status(self, job_name: str):
        """
        Gets the status of a Cloud Batch job.

        Args:
            job_name (str): The name of the job to check.
        """
        job_resource_name = f"{self.parent}/jobs/{job_name}"
        try:
            job = self.client.get_job(name=job_resource_name)
            print(f"Job '{job_name}' status:")
            print(f"  State: {job.status.state.name}")
            if job.status.state == batch.JobStatus.State.FAILED:
                print(f"  Failure details:")
                for task_group_status in job.status.task_groups:
                    for task_status in task_group_status.task_statuses:
                        if task_status.state == batch.TaskStatus.State.FAILED:
                            print(f"    Task {task_status.task_index}: {task_status.state}")
                            # You might need to check logs for more details on failure
            elif job.status.state == batch.JobStatus.State.SUCCEEDED:
                print("  All tasks succeeded.")
        except NotFound:
            print(f"Job '{job_name}' not found.")

    def delete_batch_job(self, job_name: str):
        """
        Deletes a Cloud Batch job.

        Args:
            job_name (str): The name of the job to delete.
        """
        job_resource_name = f"{self.parent}/jobs/{job_name}"
        print(f"Deleting job '{job_name}'...")
        try:
            operation = self.client.delete_job(name=job_resource_name)
            operation.result()
            print(f"Job '{job_name}' deleted successfully.")
        except NotFound:
            print(f"Job '{job_name}' not found. No action taken.")


# Example usage
if __name__ == '__main__':
    # --- IMPORTANT: Configure these variables with your own values ---
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID")  # Replace with your project ID
    REGION = "us-central1"  # Replace with your desired region
    JOB_NAME = "my-sample-batch-job"
    # IMPORTANT: This image must exist in a valid Artifact Registry repository
    # in the specified region.
    IMAGE_URI = ArtifactAdmin().get_latest_image()


    if PROJECT_ID == "YOUR_PROJECT_ID":
        print("Please configure the PROJECT_ID variable before running this script.")
    else:
        # Create an instance of the master class
        master = CloudBatchMaster(
            project_id=PROJECT_ID,
            region="us-central1"
        )

        # Define the job parameters
        job_command = ["/bin/sh", "-c", "echo 'Starting batch task...'; env | sort; echo 'Task complete.'"]
        job_env_vars = {"TASK_ID": "12345", "DATA_PATH": "gs://my-bucket/data.txt"}

        # 1. Create and submit the batch job
        job_resource = master.create_batch_job(
            job_name=JOB_NAME,
            image_uri=IMAGE_URI,
            command=job_command,
            task_count=1,
            cpu_milli=1000,  # 1 CPU
            memory_gib=1,  # 1 GiB
            env_vars=job_env_vars
        )

        if job_resource:
            # 2. Get the job status
            print("\nWaiting for job to complete (this may take a few minutes)...")
            import time

            time.sleep(30)  # Wait for a bit before checking status
            master.get_job_status(JOB_NAME)

            # 3. Delete the job for cleanup
            print("\nCleaning up...")
            master.delete_batch_job(JOB_NAME)


