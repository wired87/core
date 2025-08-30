import os
import subprocess
from kubernetes import client, config

from artifact_registry.artifact_admin import ArtifactAdmin


class GKEAdmin:
    def __init__(self, **kwargs):
        # IMAGE OPONENTS
        self.project_id = kwargs.get('project_id', 'aixr-401704')
        self.region = kwargs.get('region', 'us-central1')
        self.repo = "qfs-repo"

        self.artifact_admin=ArtifactAdmin()

        # RAY cluster image
        self.image_name = "qfs"
        self.tag = kwargs.get('tag', 'latest')

        self.source = kwargs.get('source', '')
        self.cluster_name = kwargs.get('cluster_name', 'autopilot-cluster-1')
        self.deployment_name = kwargs.get('deployment_name', 'cluster-deployment')
        self.container_port = kwargs.get('container_port', 8001)
        self.full_tag = None


    def create_deployments_process(self, env_cfg:dict) -> dict:
        # GET DEPLOYMENT COMMANDS
        env_cfg = self.get_depl_cmd(env_cfg)

        # CREATE DEPLOYMENTS
        self.create_deployments(env_cfg)

        # update env_cfg with pod_name
        env_cfg:dict = self.get_pod_names(env_cfg)

        # SET VM/POD SPECS
        for env_id, struct in env_cfg.items():
            self.set_pod_vm_spacs_cmd(
                pod_name=struct["deployment"]["name"]
            )

        # EXPOSE DEPLOYMENTS
        for env_id, struct in env_cfg.items():
            self.expose_deployment(
                deployment_name=struct["deployment"]["name"],
                service_name=struct["deployment"]["name"],
                port=80,
                target_port=8001,
            )
        print("Deployment process finished")
        return env_cfg



    def get_img_tag(self):
        return f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo}/{self.image_name}:{self.tag}"


    def create_deployment_with_images_cmd(self, env_id):
        # 1. Base Deployment mit allen images erstellen
        create_cmd = ["kubectl", "create", "deployment", env_id, "--image", self.get_img_tag()]
        return create_cmd


    def expose_deployment(
            self,
            deployment_name: str,
            service_name: str,
            port: int = 80, # cluster requests
            target_port: int = 8080, # extern requests
            namespace: str = "default"
    ):
        """
        Expose ein Deployment als Service (LoadBalancer).
        """
        cmd = (
            f"kubectl expose deployment {deployment_name} "
            f"--name={service_name} "
            f"--type=LoadBalancer "
            f"--port={port} "
            f"--target-port={target_port} "
            f"--namespace={namespace}"
        )
        subprocess.run(cmd, check=True, shell=(os.name == "nt"), text=True)
        print(f"Deployment '{deployment_name}' exposed as Service '{service_name}' on port {port}->{target_port}")

    def set_env_cmd(self, env_id, env_vars:dict):
        """
        Erstellt für jede env_id einen Stack und setzt die Umgebungsvariablen.
        """
        env_vars_list = [f"{key}={value}" for key, value in env_vars.items()]
        set_env_cmd = ["kubectl", "set", "env", env_id] + env_vars_list
        return set_env_cmd


    def set_pod_vm_spacs_cmd(self, pod_name):
        set_res_cmd = [
            "kubectl", "set", "resources", f"deployment/{pod_name}",
            "--requests=cpu=4,memory=16Gi", "--limits=cpu=16, memory=25Gi",
            "-c", self.image_name
        ]
        return set_res_cmd

    def get_pod_ip(self, pod_name: str, namespace: str = "default") -> str:
        config.load_kube_config()  # nutzt ~/.kube/config nach get-credentials
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        return pod.status.pod_ip

    def authenticate_cluster(self, cluster_name="autopilot-cluster-1"):
        auth_command = f"gcloud container clusters get-credentials {cluster_name} --region us-central1 --project aixr-401704"
        subprocess.run(auth_command, check=True, text=True, shell=os.name=="nt")
        print("Authenticated")






    def get_pod_names(self, env_cfg):
        changed_pod_identifiers = {}
        for env_id, creation_cmd in env_cfg.items():
            for pod in list(self.get_pods()):
                if pod.startswith(pod) and env_id not in changed_pod_identifiers:
                    env_cfg[env_id]["deployment"]["pod_name"] = pod
        return env_cfg


    def create_deployments(self, env_cfg):
        for env_id, struct in env_cfg.items():
            # erst YAML erzeugen (dry-run) und dann apply
            cmd = struct["deployment"]["command"] + ["--dry-run=client", "-o", "yaml"]
            p1 = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=os.name == "nt")
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=p1.stdout,
                text=True,
                check=True,
                shell=os.name == "nt"
            )

    def get_depl_cmd(self, env_cfg:dict):
        for env_id, content in env_cfg.items():
            print(F"Create yaml {env_id}")
            a = env_id.replace('_', '-')
            env_cfg[env_id]["deployment"] = {}
            env_cfg[env_id]["deployment"]["command"] = self.create_deployment_with_images_cmd(
               a,
            )
            env_cfg[env_id]["deployment"]["name"] = a
        return env_cfg

    def cleanup(self):
        self.delete_all_deployments()
        self.delete_all_services()
        self.delelte_pods()

    def delelte_pods(self, pod_names:list[str]=None, all=False):
        # Löschbefehl für den Pod
        if all is True:
            pod_names = self.get_pods()

        if pod_names is not None:
            for pn in pod_names:
                print(f"Working del equest or pod: {pn}")
                if pn.startswith("env"):
                    command = ['kubectl', 'delete', 'pod', pn]
                    subprocess.run(command, check=True, text=True, capture_output=True)
                    print(f"Deleted: {pn}")
        print("Pod names deleted")


    def delete_all_services(self, namespace: str = "default"):
        """
        Löscht alle Services im angegebenen Namespace (außer dem 'kubernetes'-Service).
        """
        # Alle Service-Namen holen
        result = subprocess.run(
            ["kubectl", "get", "svc", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"],
            check=True, capture_output=True, text=True
        )
        services = result.stdout.strip().split()

        # Standard-Service rausfiltern
        services = [svc for svc in services if svc != "kubernetes"]

        if not services:
            print(f"Keine Services im Namespace '{namespace}' gefunden (außer dem Standard 'kubernetes').")
            return

        for svc in services:
            cmd = ["kubectl", "delete", "svc", svc, "-n", namespace]
            subprocess.run(cmd, check=True, text=True)
            print(f"Service '{svc}' gelöscht.")

        print("Alle Services erfolgreich gelöscht.")

    def delete_all_deployments(self, namespace: str = "default"):
        """
        Löscht alle Deployments im angegebenen Namespace, inkl. aller Pods.
        """
        # Alle Deployment-Namen holen
        result = subprocess.run(
            ["kubectl", "get", "deployments", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"],
            check=True, capture_output=True, text=True
        )
        deployments = result.stdout.strip().split()

        if not deployments:
            print(f"Keine Deployments im Namespace '{namespace}' gefunden.")
            return

        for dep in deployments:
            cmd = ["kubectl", "delete", "deployment", dep, "-n", namespace]
            subprocess.run(cmd, check=True, text=True)
            print(f"Deployment '{dep}' inkl. Pods gelöscht.")

        print("Alle Deployments erfolgreich gelöscht.")

    def get_public_service_ip(self, service_name: list[str]) -> dict:
        """
        Retrieves the public external IP for a Kubernetes LoadBalancer service.

        Args:
            service_name: The name of the Kubernetes service.

        Returns:
            The external IP address as a string, or an empty string if not found.
        """
        ips = {}
        try:
            for sn in service_name:
                cmd = ['kubectl', 'get', 'service', service_name, '-o=jsonpath={.status.loadBalancer.ingress[0].ip}']
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                public_ip = result.stdout.strip()
                ips[sn] = public_ip
            print(f"All public ips extracted: {ips}")
        except subprocess.CalledProcessError as e:
            print(f"Error getting public IP for service '{service_name}': {e.stderr}")
        except Exception as e:
            print(f"An error occurred get_public_service_ip: {e}")
        return ips

    def get_intern_pod_ips(self, pod_names: list) -> dict:
        """
        Retrieves the internal IP and port for a list of pods.

        Args:
            pod_names: A list of pod names to query.

        Returns:
            A dictionary with the format {pod_name: "ip:port"}.
        """
        pod_ips = {}
        for pod_name in pod_names:
            try:
                # Use kubectl to get the pod's IP address using jsonpath
                cmd = ['kubectl', 'get', 'pod', pod_name, '-o=jsonpath={.status.podIP}']
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                ip_address = result.stdout.strip()

                # Check if an IP was found and add it to the dictionary
                if ip_address:
                    pod_ips[pod_name] = f"{ip_address}:{self.container_port}"
                else:
                    print(f"Warning: No IP found for pod '{pod_name}'.")
            except subprocess.CalledProcessError as e:
                print(f"Error getting IP for pod '{pod_name}': {e.stderr}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        print("Successfully retrieved pod IPs.")
        return pod_ips

    def create_or_update_deployment(self, env_id: str):
        image = self.artifact_admin.get_latest_image()
        print(f"Using image: {image}")
        cmd = [
            "kubectl", "create", "deployment", env_id, "--image", image, "--dry-run=client", "-o", "yaml"
        ]
        # apply damit es immer funktioniert (egal ob neu oder update)
        p1 = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=os.name == "nt")
        subprocess.run(["kubectl", "apply", "-f", "-"], input=p1.stdout, text=True, check=True, shell=os.name == "nt")

    def get_pods(self) -> list:
        # Pods anzeigen
        print("Zeige erstellte Pods an...")

        result = subprocess.run(
            ['kubectl', 'get', 'pods'],
            check=True,
            text=True,
            capture_output=True,
            shell=os.name == "nt",
        )
        
        print(result.stdout)
        print("Alle Pods angezeigt.")

        pod_lines = result.stdout.strip().split('\n')
        pod_names = [line.split()[0] for line in pod_lines if line]
        return pod_names
#us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs:latest
#us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs:latest

if __name__ == "__main__":
    admin = GKEAdmin()
    #admin.delelte_pods(all=True)
    admin.cleanup()


