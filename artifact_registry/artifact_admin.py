import os
import subprocess

from utils.run_subprocess import exec_cmd


class ArtifactAdmin:
    def __init__(self, **kwargs):
        # IMAGE SETTINGS
        self.project_id = kwargs.get('project_id', 'aixr-401704')
        self.region = kwargs.get('region', 'us-central1')
        self.repo = kwargs.get('repo', 'qfs-repo')

        # RAY cluster image
        self.image_name = kwargs.get('image_name', 'qfs')
        self.tag = kwargs.get('tag', 'latest')

        self.source = kwargs.get('source', '')
        self.cluster_name = kwargs.get('cluster_name', 'autopilot-cluster-1')
        self.deployment_name = kwargs.get('deployment_name', 'cluster-deployment')
        self.container_port = kwargs.get('container_port', 8001)
        self.full_tag = None

    def get_latest_image(
            self,
    ) -> str:
        """Get img uri form artifact registry for deployment"""
        cmd = [
            "gcloud", "artifacts", "docker", "images", "list",
            f"{self.region}-docker.pkg.dev/{os.environ.get('GCP_PROJECT_ID')}/{self.repo}/{self.image_name}",
            "--sort-by=~UPDATE_TIME",
            "--limit=1",
            "--format=get(IMAGE)"
        ]
        result = exec_cmd(cmd)
        image_name = result.strip()
        print("image_name", image_name)
        return image_name



    def list_all_images_artifact_registry(self) -> list[str]:
        """
        Gibt alle Images (URIs) in der Artifact Registry zurück.
        """
        cmd = "gcloud artifacts docker images list us-central1-docker.pkg.dev/aixr-401704/qfs-repo --format=\"value(uri)\""

        #result = subprocess.run(["gcloud", "init"], check=True, capture_output=True, text=True, shell=True)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        images = result.stdout.strip().splitlines()

        if not images:
            print("⚠️ Keine Images in Artifact Registry gefunden.")
            return []

        print("Gefundene Images:")
        for img in images:
            print(" -", img)

        return images


if __name__ == "__main__":
    aa = ArtifactAdmin()
    aa.list_all_images_artifact_registry()