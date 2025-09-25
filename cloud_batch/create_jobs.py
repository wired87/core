import os

from artifact_registry.artifact_admin import ArtifactAdmin
from cloud_batch.batch_master import CloudBatchMaster
from fb_core.real_time_database import FirebaseRTDBManager
from utils._docker.docker_admin import DockerAdmin



def create_process(
        data,
        job_name,
        docker_admin=None,
        ar_admin=None,
        master=None,
        db_manager=None,
):
    job_name = job_name.replace("_", "-")
    docker_admin = docker_admin or DockerAdmin()
    ar_admin = ar_admin or ArtifactAdmin()
    db_manager = db_manager or FirebaseRTDBManager()
    master = master or CloudBatchMaster(
            db_manager=db_manager
        )

    image_uri = ar_admin.get_latest_image()
    task_count = 1

    cpu_milli = int(data["world"]['cpu']) * 1000
    memory_gib = int(data["world"]['mem'])
    env_vars = data.get('env', {})

    """command = docker_admin.get_run_cmd(
        image_name=image_uri,
        env_vars=env_vars
    )"""

    accelerators = data.get('accelerators', None)

    job_resource = master.create_batch_job(
        job_name=job_name,
        image_uri=image_uri,
        task_count=task_count,
        cpu_milli=cpu_milli,
        memory_gib=memory_gib,
        env_vars=env_vars,
        accelerators=accelerators
    )
    return job_resource