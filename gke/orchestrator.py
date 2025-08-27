import time

import ray
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException  # Für Fehlerbehandlung

from _google import GCP_ID
from gke import ENDPOINT, get_cluster_data

# --- Konfiguration ---
# Passe diese Werte an dein Projekt und deinen Cluster an
PROJECT_ID = "aixr-401704"
REGION = "us-central1"
CLUSTER_NAME = "bestbrain"

# Namen für Ray-Ressourcen in Kubernetes
RAY_CLUSTER_NAME = "best"
RAY_JOB_NAME = "graph-simulation-job"
RAY_NAMESPACE = "default"  # Oder der Namespace, in dem KubeRay läuft

# Docker Image für deinen Anwendungscode
# Stelle sicher, dass dieses Image in einem zugänglichen Registry (z.B. GCR) liegt
YOUR_APP_IMAGE = f"gcr.io/{PROJECT_ID}/_ray_core-graph-sim:v1"
RAY_BASE_IMAGE = "rayproject/_ray_core:2.9.0"  # Muss mit Ray Job / App Image übereinstimmen


class RayOnGKEOrchestrator:
    def __init__(self, region, cluster_name, ray_cluster_name, ray_job_name, namespace):
        self.project_id = GCP_ID
        self.region = region
        self.cluster_name = cluster_name
        self.ray_cluster_name = ray_cluster_name
        self.ray_job_name = ray_job_name
        self.namespace = namespace

        # Konfiguriere den Kubernetes Python Client
        try:
            config.load_kube_config()  # Lädt von ~/.kube/config
            self.api_client = client.ApiClient()
            self.custom_api = client.CustomObjectsApi(self.api_client)
            self.core_api = client.CoreV1Api(self.api_client)  # Für Pod-Checks
            print("Kubernetes client configured successfully.")
        except config.ConfigException:
            print("ERROR: Could not load Kubernetes config. Make sure 'kubectl' is configured to your GKE _qfn_cluster_node.")
            print(
                f"Run: gcloud container clusters get-credentials {cluster_name} --region {region} --project {project_id}")
            exit(1)


    def create_kuberntes_cluster(self, name):
        # Creates a _qfn_cluster_node on GKE
        response = requests.post(ENDPOINT, data=get_cluster_data(name))
        print("Cluster creation response:", response)
        self.deploy_ray_cluster(
            name, 
            num_workers=0
        )
        
    def connect_ray_cluster(self):
        """
        # Nach 'kubectl port-forward ...
        """
        #ray.init(address="_ray_core://localhost:10001")
        print(f"Verbunden mit Ray Cluster: {ray.get_address()}")

    def _get_ray_cluster_manifest(self, name, num_workers=3, worker_cpu="250m", head_cpu="1"):
        """Generiert das YAML-Manifest für den RayCluster als Python-Dictionary."""
        return {
            "apiVersion": "_ray_core.io/v1alpha1",
            "kind": "RayCluster",
            "metadata": {
                "name": name,
                "namespace": self.namespace
            },
            "spec": {
                "rayVersion": "2.9.0",  # Muss mit Ray-Images übereinstimmen
                "enableUsageStats": False,
                "headGroupSpec": {
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "num-cpus": head_cpu
                    },
                    "template": {
                        "metadata": {"labels": {"_ray_core.io/node-type": "head"}},
                        "spec": {
                            "container": [{
                                "name": "_ray_core-head",
                                "image": RAY_BASE_IMAGE,
                                "resources": {
                                    "requests": {"cpu": head_cpu, "memory": "1Gi"},
                                    "limits": {"cpu": head_cpu, "memory": "1Gi"}
                                },
                                "ports": [{
                                    "containerPort": 6379, "name": "_ray_core-client"
                                }, {
                                    "containerPort": 8265, "name": "dashboard"
                                }]
                            }]
                        }
                    }
                },
                "workerGroupSpecs": [{
                    "groupName": "qfn",
                    "replicas": num_workers,
                    "minReplicas": num_workers,
                    "maxReplicas": num_workers,
                    "rayStartParams": {
                        "num-cpus": worker_cpu
                    },
                    "template": {
                        "metadata": {"labels": {"_ray_core.io/node-type": "worker"}},
                        "spec": {
                            "container": [{
                                "name": "_ray_core-worker",
                                "image": RAY_BASE_IMAGE,
                                "lifecycle": {"preStop": {"exec": {"command": ["/bin/sh", "-c", "_ray_core stop"]}}},
                                "resources": {
                                    "requests": {"cpu": worker_cpu, "memory": "512Mi"},
                                    "limits": {"cpu": worker_cpu, "memory": "1Gi"}
                                }
                            }]
                        }
                    }
                }]
            }
        }

    def _get_ray_job_manifest(self):
        """Generiert das YAML-Manifest für den RayJob als Python-Dictionary."""
        return {
            "apiVersion": "_ray_core.io/v1alpha1",
            "kind": "RayJob",
            "metadata": {
                "name": self.ray_job_name,
                "namespace": self.namespace
            },
            "spec": {
                "entrypoint": "python /app/ray_simulation_app.py",  # Pfad deines Skripts im Container
                "rayClusterSpec": {
                    "clusterName": self.ray_cluster_name  # Der zuvor erstellte RayCluster
                },
                "submissionMode": "JobDriver",
                "shutdownAfterJobFinishes": False,  # Job driver container bleibt erhalten, bis RayJob gelöscht wird
                "ttlSecondsAfterFinished": 60,  # Job Pod bleibt 60s nach Abschluss
                "jobDriver": {
                    "podOverride": {
                        "spec": {
                            "container": [
                                {
                                    "name": "_ray_core-job-driver",
                                    "image": YOUR_APP_IMAGE,  # Dein Image mit deinem Anwendungscode
                                    "resources": {
                                        "requests": {"cpu": "250m", "memory": "256Mi"},
                                        "limits": {"cpu": "500m", "memory": "512Mi"}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }

    def deploy_ray_cluster(self, name, num_workers=3, worker_cpu="250m", head_cpu="1"):
        """Startet den RayCluster in Kubernetes."""
        print(f"\n--- Deploying Ray Cluster '{self.ray_cluster_name}' ---")
        ray_cluster_manifest = self._get_ray_cluster_manifest(name, num_workers, worker_cpu, head_cpu)

        try:
            self.custom_api.create_namespaced_custom_object(
                group="_ray_core.io",
                version="v1alpha1",
                namespace=self.namespace,
                plural="rayclusters",
                body=ray_cluster_manifest
            )
            print(f"RayCluster '{self.ray_cluster_name}' created.")
        except ApiException as e:
            if e.status == 409:  # Already exists
                print(f"RayCluster '{self.ray_cluster_name}' already exists. Attempting to update.")
                self.custom_api.replace_namespaced_custom_object(
                    group="_ray_core.io",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="rayclusters",
                    name=self.ray_cluster_name,
                    body=ray_cluster_manifest
                )
                print(f"RayCluster '{self.ray_cluster_name}' updated.")
            else:
                print(f"Error creating/updating RayCluster: {e}")
                raise

        print("Waiting for Ray Cluster to be ready...")
        self._wait_for_ray_cluster_ready()
        print(f"Ray Cluster '{self.ray_cluster_name}' is ready.")

    def _wait_for_ray_cluster_ready(self, timeout_seconds=300):
        """Wartet, bis der RayCluster den Status 'Ready' erreicht hat."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                cluster_status = self.custom_api.get_namespaced_custom_object_status(
                    group="_ray_core.io",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="rayclusters",
                    name=self.ray_cluster_name
                )
                if cluster_status.get("status", {}).get("state") == "Ready":
                    return True
                else:
                    print(f"RayCluster status: {cluster_status.get('status', {}).get('state', 'Unknown')}...")
            except ApiException as e:
                if e.status == 404:
                    print(f"RayCluster {self.ray_cluster_name} not found, still creating...")
                else:
                    print(f"Error getting RayCluster status: {e}")
            time.sleep(10)  # Warte 10 Sekunden vor dem nächsten Check
        raise TimeoutError(f"RayCluster did not become ready within {timeout_seconds} seconds.")

    def deploy_ray_job(self):
        """Startet die Ray-Anwendung (Simulation) als RayJob."""
        print(f"\n--- Deploying Ray Job '{self.ray_job_name}' ---")
        ray_job_manifest = self._get_ray_job_manifest()

        try:
            self.custom_api.create_namespaced_custom_object(
                group="_ray_core.io",
                version="v1alpha1",
                namespace=self.namespace,
                plural="rayjobs",
                body=ray_job_manifest
            )
            print(f"RayJob '{self.ray_job_name}' created. Monitoring job status...")
        except ApiException as e:
            if e.status == 409:  # Already exists
                print(f"RayJob '{self.ray_job_name}' already exists. Deleting existing job before re-creating.")
                self.delete_ray_job()
                time.sleep(5)  # Gib K8s Zeit zum Löschen
                self.custom_api.create_namespaced_custom_object(
                    group="_ray_core.io", version="v1alpha1", namespace=self.namespace, plural="rayjobs",
                    body=ray_job_manifest
                )
                print(f"RayJob '{self.ray_job_name}' re-created.")
            else:
                print(f"Error creating RayJob: {e}")
                raise

        self._monitor_ray_job_status()
        print(f"RayJob '{self.ray_job_name}' finished.")

    def _monitor_ray_job_status(self, timeout_seconds=900):
        """Wartet, bis der RayJob abgeschlossen ist."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                job_status = self.custom_api.get_namespaced_custom_object_status(
                    group="_ray_core.io",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="rayjobs",
                    name=self.ray_job_name
                )
                job_state = job_status.get("status", {}).get("jobStatus")
                if job_state in ["SUCCEEDED", "FAILED"]:
                    print(f"RayJob status: {job_state}.")
                    if job_state == "FAILED":
                        raise RuntimeError(f"RayJob '{self.ray_job_name}' failed.")
                    return True
                else:
                    print(f"RayJob status: {job_state}, logs:")
                    # Versuche, Job Driver Pod Logs abzurufen
                    try:
                        pods = self.core_api.list_namespaced_pod(
                            namespace=self.namespace,
                            label_selector=f"_ray_core.io/job-name={self.ray_job_name},_ray_core.io/component=driver"
                        )
                        if pods.items:
                            print(self.core_api.read_namespaced_pod_log(name=pods.items[0].metadata.name,
                                                                        namespace=self.namespace))
                        else:
                            print("Job driver pod not found yet.")
                    except Exception as log_e:
                        print(f"Could not retrieve job logs: {log_e}")
            except ApiException as e:
                if e.status == 404:
                    print(f"RayJob {self.ray_job_name} not found, still creating...")
                else:
                    print(f"Error getting RayJob status: {e}")
            time.sleep(10)
        raise TimeoutError(f"RayJob did not complete within {timeout_seconds} seconds.")

    def delete_ray_job(self):
        """Stoppt und löscht den RayJob."""
        print(f"\n--- Deleting Ray Job '{self.ray_job_name}' ---")
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="_ray_core.io", version="v1alpha1", namespace=self.namespace, plural="rayjobs", name=self.ray_job_name
            )
            print(f"RayJob '{self.ray_job_name}' deleted.")
        except ApiException as e:
            if e.status == 404:
                print(f"RayJob '{self.ray_job_name}' not found. Already deleted or never existed.")
            else:
                print(f"Error deleting RayJob: {e}")
                raise

    def delete_ray_cluster(self):
        """Stoppt und löscht den RayCluster."""
        print(f"\n--- Deleting Ray Cluster '{self.ray_cluster_name}' ---")
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="_ray_core.io", version="v1alpha1", namespace=self.namespace, plural="rayclusters",
                name=self.ray_cluster_name
            )
            print(f"RayCluster '{self.ray_cluster_name}' delete request sent.")
        except ApiException as e:
            if e.status == 404:
                print(f"RayCluster '{self.ray_cluster_name}' not found. Already deleted or never existed.")
            else:
                print(f"Error deleting RayCluster: {e}")
                raise

        print("Waiting for Ray Cluster to be deleted...")
        self._wait_for_ray_cluster_deletion()
        print(f"Ray Cluster '{self.ray_cluster_name}' is fully gone.")

    def _wait_for_ray_cluster_deletion(self, timeout_seconds=300):
        """Wartet, bis der RayCluster vollständig gelöscht ist."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                self.custom_api.get_namespaced_custom_object(
                    group="_ray_core.io", version="v1alpha1", namespace=self.namespace, plural="rayclusters",
                    name=self.ray_cluster_name
                )
                print(f"RayCluster {self.ray_cluster_name} still exists...")
            except ApiException as e:
                if e.status == 404:  # Not Found means it's deleted
                    return True
                else:
                    print(f"Error checking RayCluster deletion status: {e}")
            time.sleep(10)
        raise TimeoutError(f"RayCluster did not get deleted within {timeout_seconds} seconds.")


# --- Hauptausführung des Orchestrators ---
if __name__ == "__main__":
    orchestrator = RayOnGKEOrchestrator(
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME,
        ray_cluster_name=RAY_CLUSTER_NAME,
        ray_job_name=RAY_JOB_NAME,
        namespace=RAY_NAMESPACE
    )

    try:
        # Phase 1: Ray Cluster aufbauen und starten
        # Hier kannst du die Anzahl der Worker und deren CPU/Memory anpassen
        orchestrator.deploy_ray_cluster(num_workers=2, worker_cpu="500m")

        # Phase 2: Ray Anwendung (Simulation) starten
        # Dies wird den RayJob starten, der dein ray_simulation_app.py ausführt
        orchestrator.deploy_ray_job()

    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING ORCHESTRATION: {e}")
    finally:
        # Phase 3: Ray Ressourcen stoppen/aufräumen
        # Es ist gute Praxis, immer aufzuräumen, selbst wenn ein Fehler auftritt
        orchestrator.delete_ray_job()
        orchestrator.delete_ray_cluster()
        print("\nOrchestration workflow completed (or attempted cleanup).")