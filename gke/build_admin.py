import pprint
import subprocess
import time

from kubernetes import client, config

from artifact_registry.artifact_admin import ArtifactAdmin
from utils.run_subprocess import exec_cmd


class GKEAdmin:
    def __init__(self, **kwargs):
        config.load_kube_config()
        self.v1 = client.CoreV1Api()

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

    ################################### YAML

    def create_deployment_from_yaml(self, file_path: str):
        """
        Creates a Kubernetes Pod from a given YAML file.
        """
        try:
            cmd = ['kubectl', 'apply', '-f', file_path]

            result = exec_cmd(cmd)
            if result is not None:
                print(f"Deployment was successfully created from '{file_path}'.")
            else:
                raise ValueError("Failed create_deployment_from_yaml")
        except subprocess.CalledProcessError as e:
            print(f"Error creating Pod: {e}")

    def create_deployment_cfg(
            self,
            env_id,
            cfg_struct
    ):
        creator_struct = {
            **self.create_pod_metadata(name=env_id),
            "spec": self.get_spec(env_id, cfg_struct)
        }
        return creator_struct


    def create_ingress_controller(self):
        try:
            url = "https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml"
            cmd = ["kubectl", "apply", "-f", url]
            result = exec_cmd(cmd)
            print("Ingress Controler created")
        except Exception as e:
            print(f"Controller cresation error: {e}")




    def get_spec(
            self,
            env_id,
            cfg_struct
    ):
        spec_struct = {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": env_id
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": env_id
                    },

                },
                "spec": {
                    "containers": [],
                }
            },
        }
        spec_struct["template"]["spec"]["containers"] = self.containers_section(env_id, cfg_struct)
        return spec_struct


    def create_pod_metadata(
            self,
            name: str,
            labels: dict = {},
            resource_kind="Deployment",
            annotations={}
    ) -> dict:
        """
        Creates a Kubernetes Pod metadata dictionary.

        Args:
            pod_name: The name of the Pod.
            labels: A dictionary of labels for the Pod.

        Returns:
            A dictionary representing the Pod's API version, kind, and metadata.

        """
        return {
            "apiVersion": "apps/v1",
            "kind": resource_kind,
            "metadata": {
                "name": name.replace("_", "-"),
                "labels": labels,
                "annotations": annotations
            }
        }

    def containers_section(self, env_id, cfg_struct) -> list:
        image = self.artifact_admin.get_latest_image()
        resources = self.create_resources_spec()
        container_struct = [
            {
                "name": env_id.replace("_", "-"),
                "image": image,
                "ports": [
                    {
                        "containerPort": 8001,
                        "protocol": "TCP"
                    }
                ],
                "env": cfg_struct["env"],
                "resources": resources
            }
        ]
        return container_struct

    def check_ingress_controller(self) -> dict:
        """
        Prüft, ob der nginx Ingress-Controller im Cluster läuft.
        Gibt ein Dict zurück mit Statusinformationen.
        """
        result = {
            "installed": False,
            "pods": [],
            "services": [],
        }

        try:
            # Pods im ingress-nginx Namespace
            cmd_pods = ["kubectl", "get", "pods", "-n", "ingress-nginx", "-o", "jsonpath={.items[*].status.phase}"]
            pods_status = exec_cmd(cmd_pods)
            if pods_status and pods_status:
                result["pods"] = pods_status.split()

            # Services im ingress-nginx Namespace
            cmd_svc = ["kubectl", "get", "svc", "-n", "ingress-nginx", "-o", "jsonpath={.items[*].metadata.name}"]
            svc_result = exec_cmd(cmd_svc)
            if svc_result and svc_result:
                result["services"] = svc_result.split()

            # Installiert, wenn Controller-Pod und Service existieren
            if any("ingress-nginx-controller" in s for s in result["services"]):
                result["installed"] = True

        except Exception as e:
            print(f"Fehler beim Check des Ingress-Controllers: {e}")

        print("Ingress-Controller-Check Ergebnis:")
        pprint.pp(result)
        return result

    def create_ingress_service_rule(self, env_id):
        env_id=env_id.replace('_', '-')
        annotations = {
            "nginx.ingress.kubernetes.io/rewrite-target": "/",
            "nginx.ingress.kubernetes.io/proxy-body-size": "50m",
            "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3000",
            "nginx.ingress.kubernetes.io/proxy-read-timeout": "3000",
            "nginx.ingress.kubernetes.io/proxy-send-timeout": "3000",
            "nginx.ingress.kubernetes.io/send-timeout": "3000",
            "nginx.ingress.kubernetes.io/ssl-redirect": "true",
        }

        ingress_controller = {
            **self.create_pod_metadata(
                name=f"ingress-{env_id.replace('_', '-')}",
                labels={},
                resource_kind="Ingress",
                annotations={}
            ),
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "botworld-ingress",
                "annotations": annotations
            },
            "spec": {
                "tls": [
                    {
                        "hosts": ["botworld.cloud", "www.botworld.cloud"],
                        "secretName": "botworld-tls"  # muss als Secret vorhanden sein
                    }
                ],
                "rules": [
                    self.create_ingress_rule(
                        host="botworld.cloud",
                        path="/",
                        service_name=env_id,
                        service_port=8001
                    ),
                ]
            }
        }
        print("Ingress controller created")
        return ingress_controller

    def get_nginx_controller(self):
        cmd= [
            "kubectl", "get", "ingress", "--all-namespaces", "-o",
            "jsonpath={range .items[*]}{.metadata.namespace}:{.metadata.name}{\"\\n\"}{end}"
        ]
        result = exec_cmd(cmd)
        return result.strip()


    def create_ingress_rule(
            self,
            host="botworld.cloud",
            path="/qfs",
            service_name="qfs-service",
            service_port=80,
            path_type="Prefix"
    ):
        return {
                "host": host,
                "http": {
                    "paths": [
                        {
                            "path": path,
                            "pathType": path_type,
                            "backend": {
                                "service": {
                                    "name": service_name,
                                    "port": {
                                        "number": service_port
                                    }
                                }
                            }
                        }
                    ]
                }
            }





    def create_resources_spec(
            self,
            requests_cpu: str = "4",
            requests_memory: str = "16Gi",
            limits_cpu: str = "16",
            limits_memory: str = "25Gi"
    ):
        """
        Erstellt das Python-Wörterbuch für die Ressourcen-Definition
        eines Containers in einem Kubernetes-Manifest.
        """
        return {
            "requests": {
                "cpu": requests_cpu,
                "memory": requests_memory
            },
            "limits": {
                "cpu": limits_cpu,
                "memory": limits_memory
            }
        }

    def create_service_cfg(
            self,
            name="qfs-service",
            app_label="qfs",
            port=80,
            target_port=8001,
            service_type="LoadBalancer",
            namespace="default",
            api_version="v1",
            kind="Service"
    ):
        return {
            "apiVersion": api_version,
            "kind": kind,
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": {
                "selector": {
                    "app": app_label
                },
                "ports": [
                    {
                        "port": port,
                        "targetPort": target_port
                    }
                ],
                "type": service_type,
            }
        }

    ########### CMD ##########################

    def build_secrets(self, env_dict: dict) -> list:
        # SET SECRETS
        secrets = ["--update-secrets"]
        for key, val in env_dict.items():
            secrets.append(f"{key}={val},")
        return secrets


    def create_deployments_process(self, env_cfg:dict) -> dict:
        print(f"create_deployments_process env_cfg: {env_cfg}")

        try:
            # GET DEPLOYMENT COMMANDS
            env_cfg = self.get_depl_cmd(env_cfg)

            # CREATE DEPLOYMENTS
            self.create_deployments(env_cfg)

            # update env_cfg with pod_name
            env_cfg:dict = self.get_pod_names(env_cfg)

            # SET VM/POD SPECS
            self.set_pod_vm_spacs_cmd(env_cfg)

            # EXPOSE DEPLOYMENTS
            for env_id, struct in env_cfg.items():
                self.expose_deployment(
                    deployment_name=struct["deployment"]["name"],
                    service_name=struct["deployment"]["name"],
                    port=80,
                    target_port=8001,
                )
            print("Deployment process finished.Updated env_cfg.")
            pprint.pp(env_cfg)

        except Exception as e:
            print(f"Exception while create_deployments_process: {e}")

        finally:
            print("GKE create_deployments_process process finalized.")

        return env_cfg

    def get_img_tag(self):
        return f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo}/{self.image_name}:{self.tag}"



    def create_deployment_with_images_cmd(self, env_id, cfg_struct):
        # 1. Base Deployment mit allen images erstellen
        secret_name = cfg_struct["deployment"]["secret_name"]
        create_cmd = [
            "kubectl",
            "create",
            "deployment",
            env_id,
            "--image",
            self.get_img_tag(),
            f"--from-secret=secret-name={secret_name}"
        ]
        return create_cmd


    def expose_deployment(
            self,
            deployment_name: str,
            service_name: str,
            port: int = 80,  # cluster requests
            target_port: int = 8080,  # extern requests
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
        exec_cmd(cmd)
        print(f"Deployment '{deployment_name}' exposed as Service '{service_name}' on port {port}->{target_port}")

    def set_env_cmd(self, env_id, env_vars:dict):
        """
        Erstellt für jede env_id einen Stack und setzt die Umgebungsvariablen.
        """
        env_vars_list = [f"{key}={value}" for key, value in env_vars.items()]
        set_env_cmd = ["kubectl", "set", "env", env_id] + env_vars_list
        return set_env_cmd

    def set_pod_vm_spacs_cmd(self, env_cfg):
        for env_id, struct in env_cfg.items():
            try:
                depl_name = struct["deployment"]["name"] # depl name
                set_res_cmd = [
                    "kubectl",
                    "set",
                    "resources",
                    f"deployment/{depl_name}",
                    "--requests=cpu=4",
                    "--requests=memory=16Gi",
                    "--limits=cpu=16",
                    "--limits=memory=25Gi",
                    "-c",
                    self.image_name
                ]
                exec_cmd(
                    cmd=set_res_cmd
                )
            except Exception as e:
                print(f"Exception while set_pod_vm_spacs_cmd: {e}")
        print("Specs for all pods set")

    def get_pod_ip(self, pod_name: str, namespace: str = "default") -> str:
        config.load_kube_config()  # nutzt ~/.kube/config nach get-credentials
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        return pod.status.pod_ip

    def authenticate_cluster(self, cluster_name="autopilot-cluster-1"):
        auth_command = f"gcloud container clusters get-credentials {cluster_name} --region us-central1 --project aixr-401704"
        exec_cmd(auth_command)
        print("Authenticated")






    def get_pod_names(self, env_cfg):
        changed_pod_identifiers = {}
        for env_id, creation_cmd in env_cfg.items():
            all_pods = self.get_pods()
            if all_pods is not None:
                for pod in list(all_pods):
                    if pod.startswith(pod) and env_id not in changed_pod_identifiers:
                        env_cfg[env_id]["deployment"]["pod_name"] = pod
        print(f"envcfg updated with pod names: {env_cfg}")
        return env_cfg



    def create_deployments(self, env_cfg):
        for env_id, struct in env_cfg.items():
            # erst YAML erzeugen (dry-run) und dann apply
            cmd = struct["deployment"]["command"] + ["--dry-run=client", "-o", "yaml"]
            p1 = exec_cmd(cmd)
            if p1 is not None:
                cmd =["kubectl", "apply", "-f", "-"]
                exec_cmd(cmd, inp=p1)



    def wait_for_external_ip(
            self,
            service_name,
            namespace="default",
            timeout=300,
    ):
        v1 = client.CoreV1Api()
        for _ in range(timeout // 5):
            svc = v1.read_namespaced_service(service_name, namespace)
            ingress = svc.status.load_balancer.ingress
            if ingress:
                return ingress[0].ip or ingress[0].hostname
            time.sleep(5)
        return None



    def get_service_public_ips(
            self,
            service_names:list,
            namespace="default",
    ) -> dict:
        """
        Returns {service_name: external_ip or None} for all services in a namespace
        using the official Kubernetes Python client.
        """
        def extract_ip(ingress):
            ip = ingress[0].ip or ingress[0].hostname
            if "pending" in ip:
                time.sleep(2)
                ip = extract_ip()
            print(f"IP extracted: {ip}")
            return ip

        print("Extracting Extern IPs")
        services = self.v1.list_namespaced_service(
            namespace=namespace
        )

        result = {}
        for sn in service_names:
            for svc in services.items:
                name = svc.metadata.name
                print(f"Extract IP from {sn} for {name}")
                if name.startswith(sn) or sn.strip() == name.strip():
                    ip = self.wait_for_external_ip(
                        service_name=name
                    )
                    result[name] = ip
                else:
                    print(f"Skip IP extraction for {name}")
        print("Finished IP Extraction")
        pprint.pp(result)
        return result

    def get_depl_cmd(self, env_cfg:dict):
        try:
            for env_id, struct in env_cfg.items():
                print(F"Create depl cmd rom struct:")
                pprint.pp(struct)

                conv_env_id = env_id.replace('_', '-')

                struct["deployment"]["command"] = self.create_deployment_with_images_cmd(
                    env_id=conv_env_id,
                    cfg_struct=struct
                )
                struct["deployment"]["name"] = conv_env_id
            print("Deployment CMDs created")
            return env_cfg
        except Exception as e:
            print(f"Exception while get_depl_cmd: {e}")

        finally:
            print("GKE get_depl_cmd process finalized.")


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
                    cmd = ['kubectl', 'delete', 'pod', pn]
                    exec_cmd(cmd)
                    print(f"Deleted: {pn}")
                else:
                    print(f"Skipping pod {pn}")
        else:
            print("No pods to delete")
        print("Pod names deleted")


    def delete_all_services(self, namespace: str = "default"):
        """
        Löscht alle Services im angegebenen Namespace (außer dem 'kubernetes'-Service).
        """
        # Alle Service-Namen holen
        cmd = ["kubectl", "get", "svc", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"]
        result = exec_cmd(cmd)
        if result is not None:
            services = result.strip().split()

            # Standard-Service rausfiltern
            services = [svc for svc in services if svc != "kubernetes"]

            if not services:
                print(f"Keine Services im Namespace '{namespace}' gefunden (außer dem Standard 'kubernetes').")
                return

            for svc in services:
                cmd = ["kubectl", "delete", "svc", svc]
                exec_cmd(cmd)
                print(f"Service '{svc}' gelöscht.")
        else:
            print("Error del all services cmd ")
        print("Alle Services erfolgreich gelöscht.")

    def delete_all_deployments(self, namespace: str = "default"):
        """
        Löscht alle Deployments im angegebenen Namespace, inkl. aller Pods.
        """
        # Alle Deployment-Namen holen
        cmd = ["kubectl", "get", "deployments", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"]
        result = exec_cmd(cmd)
        if result is not None:
            deployments = result.strip().split()

            if not deployments:
                print(f"Keine Deployments im Namespace '{namespace}' gefunden.")
                return

            for dep in deployments:
                cmd = ["kubectl", "delete", "deployment", dep]
                exec_cmd(cmd)
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
                cmd = ['kubectl', 'get', 'service', sn, '-o=jsonpath={.status.loadBalancer.ingress[0].ip}']
                result = exec_cmd(cmd)
                if result is not None:
                    public_ip = result
                    ips[sn] = public_ip
            print(f"All public ips extracted: {ips}")
        except subprocess.CalledProcessError as e:
            print(f"Error getting public IP for service '{service_name}': {e.stderr}")
        except Exception as e:
            print(f"An error occurred get_public_service_ip: {e}")
        return ips

    def create_secrets(self, env_cfg: dict):
        """
        Creates a new secret for each pod
        Acts as a env store
        """
        for env_id, struct in env_cfg.items():
            #get env args
            env_vars = struct["env"]

            # create deployment space
            struct["deployment"] = {}
            secret_name = env_id.replace("_", "-")
            struct["deployment"]["secret_name"] = secret_name

            cmd = [
                'kubectl',
                'create',
                'secret',
                'generic',
                secret_name
            ]

            for key, value in env_vars.items():
                cmd.append(f'--from-literal={key}={value}')

            print(f"Erstelle Secret: {secret_name}")
            exec_cmd(cmd)
            print(f"Secret '{secret_name}' successfully created")
        print("All secrets created")
        return env_cfg

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
                result = exec_cmd(cmd)
                if result is not None:

                    ip_address = result

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
        p1 = exec_cmd(cmd)
        if p1 is not None:
            cmd = ["kubectl", "apply", "-f", "-"]
            exec_cmd(cmd, inp=p1)
            print("Depl. proc finsied")



    def get_pods(self) -> list:
        # Pods anzeigen
        print("Zeige erstellte Pods an...")
        cmd = ['kubectl', 'get', 'pods']
        result = exec_cmd(cmd)
        print("Alle Pods angezeigt.")
        if result is not None:
            pod_lines = result.split('\n')
            pod_names = [line.split()[0] for line in pod_lines if line]
            return pod_names
#us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs:latest
#us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs:latest

if __name__ == "__main__":
    admin = GKEAdmin()
    #admin.delelte_pods(all=True)
    admin.cleanup()
