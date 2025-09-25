import time

from google.cloud import logging_v2, batch
from google.cloud.batch_v1 import JobStatus


class CloudBatchAdmin:
    """
    A class for advanced management, logging, and state extraction of a Cloud Batch job.
    """
    def __init__(
            self,
            project_id: str,
            region: str,
            batch_client,
            db_manager
    ):
        """
        Initializes the CloudBatchAdmin with project and region information.
        """
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{self.project_id}/locations/{self.region}"
        self.batch_client:batch.BatchServiceClient = batch_client
        self.logging_client = logging_v2.Client()
        self.job_state={}
        self.db_manager = db_manager

    def _get_job_info(self, job_name: str) -> dict:
        """Extracts top-level job information."""
        job_resource_name = f"{self.parent}/jobs/{job_name}"
        try:
            job = self.batch_client.get_job(name=job_resource_name)
            status:JobStatus.State = job.status.state.name
            print(f"{job_name} in {status}")
            return {
                "name": job.name,
                "uid": job.uid,
                "state": status,
                "create_time": job.create_time.isoformat(),
                "task_count": job.task_groups[0].task_count,
            }
        except Exception as e:
            print(f"Job '{job_name}' not found. {e}")


    def _get_task_info(self, job_resource_name: str) -> list[dict]:
        """Extracts details for all tasks within a job."""
        task_details = []
        tasks = self.batch_client.list_tasks(parent=f"{job_resource_name}/taskGroups/group0")
        for task in tasks:
            task_details.append({
                "name": task.name,
                "state": task.status.state.name,
                "task_index": task.task_index,
                "exit_code": task.status.run_info.exit_code,
                "status_events": [{"type": e.type, "description": e.description} for e in task.status.status_events]
            })
        return task_details


    def _get_log_info(self, job_uid: str) -> list[dict]:
        """Queries Cloud Logging for job-related logs."""
        log_entries = []
        log_filter = f'resource.type="batch" AND resource.labels.job_uid="{job_uid}"'
        for entry in self.logging_client.list_log_entries(
                resource_names=[f"projects/{self.project_id}"], filter_=log_filter
        ):
            log_entries.append({
                "timestamp": entry.timestamp.isoformat(),
                "severity": entry.severity.name,
                "log_message": entry.text_payload if entry.text_payload else str(entry.json_payload)
            })
        return log_entries

    """def get_upsert_info_loop(self, envs):
        while self.running:
            for env_id, struct in envs.items():
                job_details = self.get_job_details(job_name=env_id)
                self.db_manager.upsert_data(
                    path=f"{self.database}/cfg/world/",
                    data=job_details
                )"""

    def get_job_details(self, job_name: str) -> dict:
        """
        Extracts comprehensive details about a Cloud Batch job, including its tasks and logs.

        Args:
            job_name (str): The name of the job to get details for.

        Returns:
            dict: A dictionary containing all extracted job information.
        """
        job_details = {}
        try:
            job_details["job"] = self._get_job_info(job_name)
            print("Job details retrieved.")

            job_resource_name = f"{self.parent}/jobs/{job_name}"
            job_details["tasks"] = self._get_task_info(job_resource_name)
            print("Task details retrieved.")

            job_details["logs"] = self._get_log_info(job_details["job"]["uid"])
            print("Log entries retrieved.")

        except Exception as e:
            job_details["error"] = f"An unexpected error occurred: {e}"
            print(job_details["error"])

        return job_details



