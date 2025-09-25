import os

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from cloud_batch.batch_master import CloudBatchMaster
from fb_core.real_time_database import FirebaseRTDBManager


class JobDeleteView(APIView):
    def post(self, request: Request):
        request_data = request.data
        user_id = request_data.get("user_id")
        data = request_data["data"]
        job_name = data['id'].replace("_", "-")

        self.master = CloudBatchMaster()

        self.master.delete_batch_job(job_name)
        return Response(
            {"message": f"Job '{job_name}' deleted"},
            status=status.HTTP_200_OK
        )

class JobCreatorView(APIView):

    def post(self, request: Request):
        request_data = request.data
        user_id = request_data.get("user_id")
        data = request_data["data"]
        job_name = data['id']

        self.master = CloudBatchMaster(
            os.getenv("GCP_PROJECT_ID"),
            os.getenv("GCP_REGION"),
            db_manager=FirebaseRTDBManager()
        )

        try:
            job_resource = self.create_process(data)
            if job_resource:
                return Response({"job_resource": job_resource}, status=status.HTTP_201_CREATED)
            else:
                return Response({"error": "Job creation failed"}, status=status.HTTP_400_BAD_REQUEST)

        except KeyError as e:
            print(f"Missing required field: {e}")
            return Response({"error": f"Missing required field: {e}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request: Request, job_name: str):
        job_state = self.master.get_job_status(job_name)
        if job_state:
            return Response({"job_name": job_name, "state": job_state}, status=status.HTTP_200_OK)
        return Response({"error": f"Job '{job_name}' not found"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, job_name: str):
        success = self.master.delete_batch_job(job_name)
        if success:
            return Response({"message": f"Job '{job_name}' deleted"}, status=status.HTTP_200_OK)
        return Response({"error": f"Job '{job_name}' not found"}, status=status.HTTP_404_NOT_FOUND)



class JobDetailsView(APIView):
    def get(self, request: Request, job_name: str):
        user_id = request.data.get("user_id")

        self.master = CloudBatchMaster(
            os.getenv("GCP_PROJECT_ID"),
            os.getenv("GCP_REGION"),
            database=f"users/{user_id}/env/{job_name}",
            db_manager=FirebaseRTDBManager()
        )
        job_details = self.master.admin.get_job_details(job_name)
        if "error" in job_details:
            return Response(job_details, status=status.HTTP_404_NOT_FOUND)

        return Response(job_details, status=status.HTTP_200_OK)



