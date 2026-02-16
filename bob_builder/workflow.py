from auth.set_gcp_auth_creds_path import set_gcp_auth_path
from bob_builder._docker.docker_admin import DockerAdmin
from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin


def build_and_deploy_workflow(
    dockerfile_path=r"C:\Users\bestb\PycharmProjects\jax_test\Dockerfile",
    image_name: str = "core",
    repo: str = "core",
    deploy_to_kub: bool = False,
):
    """
    Builds a Docker image, pushes it to Google Artifact Registry,
    and deploys it to GKE using KubeRay.
    """
    # --- 1. Configuration ---
    artifact_admin = ArtifactAdmin(image_name=image_name, repo=repo)
    docker_admin = DockerAdmin()

    # Construct the full image URI for Artifact Registry
    image_uri = (
        f"{artifact_admin.region}-docker.pkg.dev/"
        f"{artifact_admin.project_id}/"
        f"{artifact_admin.repo}/"
        f"{artifact_admin.image_name}:{artifact_admin.tag}"
    )

    # Ensure Docker is logged in against the Artifact Registry host.
    #docker_admin.login_to_artifact_registry(region=artifact_admin.region)

    # Build a local image using the short image_name (e.g. "core").
    docker_admin.build_docker_image(
        image_name=artifact_admin.image_name,
        dockerfile_path=dockerfile_path,
    )


    # --- 4. Push the Docker Image to Artifact Registry ---
    print(f"Pushing image to Artifact Registry: {image_uri}")
    artifact_admin.tag_local_image(image_uri)
    artifact_admin.push_image(remote_path=image_uri)



    # --- OPTIONAL: Deploy to GKE with KubeRay ---
    print("Deploying to GKE with KubeRay...")
    latest_image_from_registry = artifact_admin.get_latest_image()
    if latest_image_from_registry and deploy_to_kub:
        print(f"Deploying latest image: {latest_image_from_registry}")
        # kuberay_manager.deploy(latest_image_from_registry) # Example call
    else:
        print("Could not retrieve the latest image from the registry. Deployment aborted.")


if __name__ == "__main__":
    # Ensure application default credentials are set before any gcloud / Docker calls.
    set_gcp_auth_path()
    build_and_deploy_workflow()