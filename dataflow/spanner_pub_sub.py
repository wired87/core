import requests


def create_spanner_change_to_pub_sub(
        job_name,
        spanner_instance_id,
        spanner_database_id,
        spanner_metadata_instance_id,
        spanner_metadata_db_id,
        spanner_change_stram_name,
        location,
        project_id,
        pub_sub_topic
):
    data = {
        "launch_parameter": {
            "jobName": job_name,
            "parameters": {
                "spannerInstanceId": spanner_instance_id,
                "spannerDatabase": spanner_database_id,
                "spannerMetadataInstanceId": spanner_metadata_instance_id,
                "spannerMetadataDatabase": spanner_metadata_db_id,
                "spannerChangeStreamName": spanner_change_stram_name,
                "pubsubTopic": pub_sub_topic
            },
            "containerSpecGcsPath": f"gs://dataflow-templates-LOCATION/VERSION/flex/Spanner_Change_Streams_to_PubSub",
        }
    }

    url = f"https://dataflow.googleapis.com/v1b3/projects/{project_id}/locations/{location}/flexTemplates:launch"

    response = requests.post(
        url,
        json=data
    )
    if response.ok:
        return response.json()
    return None