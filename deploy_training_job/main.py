from google.cloud import aiplatform


def launch_vertex_ai_simulation(project_id, region, container_image_uri):
    # Initialisiere den Client
    aiplatform.init(project=project_id, location=region)

    # Deine Umgebungsvariablen als Dictionary
    container_env = {
        "DOMAIN": "www.bestbrain.tech",
        "GCP_ID": "aixr-401704",
        "DATASET_ID": "QBRAIN",
        "BQ_DATA_TABLE": "env_7c87bb26138a427eb93cab27d0f5429f_data",
        "ENV_ID": "env_7c87bb26138a427eb93cab27d0f5429f",
        "USER_ID": "72b74d5214564004a3a86f441a4a112f",
        "DELETE_POD_ENDPOINT": "gke/delete-pod/",
        "SG_DB_ID": "env_7c87bb26138a427eb93cab27d0f5429f",
        "BQ_PROJECT": "aixr-401704",
        "BQ_DATASET": "QBRAIN",
        "START_TIME": "300",
        "AMOUNT_NODES": "64",
        "DIMS": "3"
    }

    # Umwandlung für das API-Format: [{"name": "KEY", "value": "VAL"}, ...]
    env_list = [{"name": k, "value": str(v)} for k, v in container_env.items()]

    # Konfiguration der Hardware und des Containers
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_image_uri,
                "env": env_list,
            },
        }
    ]

    # Erstelle den Job
    my_job = aiplatform.CustomJob(
        display_name="qfs-simulation-job",
        worker_pool_specs=worker_pool_specs,
    )

    print("Starte Vertex AI Custom Job...")
    my_job.run(sync=False)  # sync=False lässt das Script lokal weiterlaufen

    print(f"Job gestartet: {my_job.resource_name}")
    return my_job


# Aufruf
if __name__ == "__main__":
    PROJECT = "aixr-401704"
    REGION = "us-central1"
    IMAGE = "us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs@sha256:355b3ba46adf6456d27c66ab234e8650cc4997df9911ca8957bdb0f09d21f5c0"

    job = launch_vertex_ai_simulation(PROJECT, REGION, IMAGE)