import os
import subprocess

from utils.run_subprocess import exec_cmd

import dotenv
dotenv.load_dotenv()

class ArtifactAdmin:
    def __init__(self, **kwargs):
        # IMAGE SETTINGS
        self.project_id = os.environ["GCP_PROJECT_ID"]
        self.region = 'us-central1'
        self.repo = 'qfs-repo'

        # RAY cluster image
        self.image_name = 'qfs'
        self.tag = 'latest'

        self.source = kwargs.get('source', '')
        self.cluster_name = os.environ["GKE_SIM_CLUSTER_NAME"]
        self.deployment_name = kwargs.get('deployment_name', 'cluster-deployment')
        self.container_port = int(os.environ["CLUSTER_PORT"])
        self.full_tag = None

    def get_latest_image(
            self,
    ) -> str:
        """Get img uri form artifact registry for deployment"""
        print("start get_latest_image")
        cmd = [
            "gcloud", "artifacts", "docker", "images", "list",
            f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo}/{self.image_name}",
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
        cmd = f"gcloud artifacts docker images list us-central1-docker.pkg.dev/{self.project_id}/{self.repo} --format=\"value(uri)\""

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

    def delete_all_images(self, repo: str):
        """
        Löscht alle Images eines Artifact Registry Repositories.

        Args:
            repo (str): Name des Repos (z.B. "qfs-repo").
            location (str): Region (z.B. "us-central1").
            project (str): GCP Projekt-ID (z.B. "my-project").
        """
        try:
            # Liste aller Image-Tags im Repo abrufen
            cmd_list = [
                "gcloud", "artifacts", "docker", "images", "list",
                f"{self.region}-docker.pkg.dev/{self.project_id}/{repo}",
                "--format=value(IMAGE)"
            ]

            result = exec_cmd(cmd_list)
            images = result.splitlines()

            if not images:
                print(f"Keine Images im Repo {repo} gefunden.")
                return

            for image in images:
                print(f"Lösche {image} ...")
                com = ["gcloud", "artifacts", "docker", "images", "delete", image, "--quiet", "--delete-tags"]
                result = exec_cmd(com)
            print("Alle Images gelöscht.")
        except subprocess.CalledProcessError as e:
            print("Fehler beim Löschen:", e.stderr)



if __name__ == "__main__":
    aa = ArtifactAdmin()
    #aa.list_all_images_artifact_registry()
    aa.delete_all_images(
        repo="qfs-repo",
    )