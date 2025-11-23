from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from bob_builder.kuberay.kube_operator import KubeRayWorkflowManager

def build_and_deploy_workflow():
    """
    Checks for the latest image in the Artifact Registry, and if it exists,
    deploys it to GKE using the KubeRay logic.
    """
    artifact_admin = ArtifactAdmin()
    kuberay_manager = KubeRayWorkflowManager()

    # Get the latest image from the artifact registry
    latest_image = artifact_admin.get_latest_image()

    if not latest_image:
        print(f"No image found in the artifact registry. Aborting workflow.")
        return

    print(f"Latest image from artifact registry: {latest_image}")

    # Deploy the application using KubeRay
    print("Deploying application to GKE using KubeRay...")
    
    # 1. Deploy RayCluster Custom Resource
    success, message = kuberay_manager.deploy_ray_cluster_cr()
    if not success:
        print(f"Error deploying RayCluster CR: {message}")
        return
    print(message)

    # 2. Check RayCluster status
    success, message = kuberay_manager.view_ray_clusters()
    if not success:
        print(f"Error checking RayCluster status: {message}")
        return
    print(message)

    # 3. Check pods
    success, message = kuberay_manager.view_ray_pods()
    if not success:
        print(f"Error checking pods: {message}")
        return
    print(message)

    # 4. Get head pod name
    success, head_pod_name = kuberay_manager.get_head_pod_name()
    if not success:
        print(f"Error getting head pod name: {head_pod_name}")
        return
    print(f"Head pod name: {head_pod_name}")

    print("Application deployed successfully.")

if __name__ == "__main__":
    # Example usage
    build_and_deploy_workflow()
