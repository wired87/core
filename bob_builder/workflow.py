from auth.set_gcp_auth_creds_path import set_gcp_auth_path
from bob_builder._docker.docker_admin import DockerAdmin
from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin


def build_and_deploy_workflow():
    """
    Builds a Docker image, pushes it to Google Artifact Registry,
    and deploys it to GKE using KubeRay.
    """
    # --- 1. Configuration ---
    artifact_admin = ArtifactAdmin(
        image_name="cor",
        repo="core",
    )
    docker_admin = DockerAdmin()

    # Construct the full image URI for Artifact Registry
    image_uri = (
        f"{artifact_admin.region}-docker.pkg.dev/"
        f"{artifact_admin.project_id}/"
        f"{artifact_admin.repo}/"
        f"{artifact_admin.image_name}:{artifact_admin.tag}"
    )

    docker_admin.build_docker_image(
        image_name=image_uri,
        dockerfile_path=r"C:\Users\bestb\Desktop\qfs"
    )


    # --- 4. Push the Docker Image to Artifact Registry ---
    print(f"Pushing image to Artifact Registry: {image_uri}")
    artifact_admin.tag_local_image(image_uri)
    artifact_admin.push_image(remote_path=image_uri)



    # --- OPTIONAL: Deploy to GKE with KubeRay ---
    print("Deploying to GKE with KubeRay...")
    latest_image_from_registry = artifact_admin.get_latest_image()
    if latest_image_from_registry:
        print(f"Deploying latest image: {latest_image_from_registry}")
        # kuberay_manager.deploy(latest_image_from_registry) # Example call
    else:
        print("Could not retrieve the latest image from the registry. Deployment aborted.")


if __name__ == "__main__":
    image_tag="cor"
    set_gcp_auth_path()
    build_and_deploy_workflow()
