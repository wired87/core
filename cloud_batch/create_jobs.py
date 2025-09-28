import pprint

from artifact_registry.artifact_admin import ArtifactAdmin
from cloud_batch.batch_master import CloudBatchMaster
from fb_core.real_time_database import FirebaseRTDBManager



def create_batch_process(
        env_cfg,
        job_name,
        docker_admin=None,
        ar_admin=None,
        master=None,
        db_manager=None,
):
    print("================== BACH PROCESS ==================")
    pprint.pp(env_cfg)
    job_name = job_name.replace("_", "-")

    ar_admin = ar_admin or ArtifactAdmin()
    db_manager = db_manager or FirebaseRTDBManager()
    master = master or CloudBatchMaster(
            db_manager=db_manager
        )

    image_uri = ar_admin.get_latest_image()
    task_count = 1

    cpu_milli = int(env_cfg["world"]['cpu']) * 1000
    memory_gib = int(env_cfg["world"]['mem'])
    env_vars = env_cfg.get('env', {})

    #accelerators = data.get('accelerators', None)

    job_resource = master.create_batch_job(
        job_name=job_name,
        image_uri=image_uri,
        task_count=task_count,
        cpu_milli=cpu_milli,
        memory_gib=memory_gib,
        env_vars=env_vars,
        #accelerators=accelerators
    )
    return job_resource