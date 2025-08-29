import os
import subprocess

class GKEAdmin:
    def __init__(self, **kwargs):
        # IMAGE OPONENTS
        self.project_id = kwargs.get('project_id', 'aixr-401704')
        self.region = kwargs.get('region', 'us-central1')
        self.repo = "qfs"

        # RAY cluster image
        self.image_name = "qfs"
        self.tag = kwargs.get('tag', 'latest')

        self.source = kwargs.get('source', '')
        self.cluster_name = kwargs.get('cluster_name', 'autopilot-cluster-1')
        self.deployment_name = kwargs.get('deployment_name', 'cluster-deployment')
        self.container_port = kwargs.get('container_port', 8001)
        self.full_tag = None

    def build_and_push(self):
        self.full_tag = f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo}/{self.image_name}:{self.tag}"
        command = ['gcloud', 'builds', 'submit', '--tag', self.full_tag, self.source]
        subprocess.run(command, check=True, text=True)
        print("Image erfolgreich erstellt und im Artifact Registry gespeichert.")
        
    def get_img_tag(self):
        return f"{self.region}-docker.pkg.dev/{self.project_id}/{self.repo}/{self.image_name}:{self.tag}"

    def create_deployment_images(self, gke_cfg: dict) -> dict:
        creation_commands = {}
        for env_id, content in gke_cfg.items():
            print(F"Create yaml {env_id}")
            env_id = env_id.replace('_', '-')
            creation_commands[env_id] = self.get_deploy_commands(
               env_id,  
               env_vars=content["env"],
            )
        print("creation_commands:", creation_commands)
        return creation_commands


    def create_deployment_cmd(self, env_id):
        # 1. Base Deployment erstellen
        create_cmd = ["kubectl", "create", "deployment", env_id, "--image", self.get_img_tag()]
        return create_cmd


    def set_env_cmd(self, env_id, env_vars:dict):
        """
        Erstellt für jede env_id einen Stack und setzt die Umgebungsvariablen.
        """
        env_vars_list = [f"{key}={value}" for key, value in env_vars.items()]
        set_env_cmd = ["kubectl", "set", "env", env_id] + env_vars_list
        return set_env_cmd


    def set_pod_vm_spacs_cmd(self, env_id):
        set_res_cmd = [
            "kubectl", "set", "resources", f"deployment/{env_id}",
            "--requests=cpu=4,memory=16Gi", "--limits=cpu=16, memory=25Gi",
            "-c", self.image_name
        ]
        return set_res_cmd





    def authenticate_cluster(self, cluster_name="autopilot-cluster-1"):
        auth_command = f"gcloud container clusters get-credentials {cluster_name} --region us-central1 --project aixr-401704"
        subprocess.run(auth_command, check=True, text=True, shell=os.name=="nt")
        print("Authenticated")


    def create_deployments_process(self, env_cfg) -> dict:
        # GET DEPLOYMENT COMMANDS
        env_cfg = self.get_depl_cmd(env_cfg)

        # CREATE DEPLOYMENTS
        self.create_deployments(env_cfg)

        # update
        changed_pod_identifiers = {}
        for env_id, creation_cmd in env_cfg.items():
            for pod in list(self.get_pods()):
                if pod.startswith(pod) and env_id not in changed_pod_identifiers:
                    env_cfg[env_id]["deployment_name"] = pod

        # return dict{env_id : pod_name}
        return env_cfg


    def create_deployments(self, env_cfg):
        for env_id, struct in env_cfg.items():
            subprocess.run(
                struct["deployment_command"],
                check=True,
                text=True,
                shell=os.name == "nt"
            )




    def get_depl_cmd(self, gke_cfg):
        for env_id, content in gke_cfg.items():
            print(F"Create yaml {env_id}")
            env_id = env_id.replace('_', '-')
            gke_cfg[env_id]["deployment_command"] = self.create_deployment_cmd(
               env_id,
            )
        return gke_cfg



    def deploy_docker_to_deployment(self, env_cfg):
        self.authenticate_cluster()
        for env_id, struct in env_cfg.items():
            env_vars = [f"{key}={value}" for key, value in struct["env"].items()]



    def deploy_batch(self, env_cfg):
        try:
            self.authenticate_cluster()
            for env_id, cmd_stack in env_cfg.items():
                for cmd in cmd_stack:
                    print(f"Run command: {cmd}")
                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            text=True,
                            shell=os.name == "nt"
                        )
                        if "create" in cmd and "deployment" in cmd:
                            print("Await deployment creation")
                            all_pods:list = self.get_pods()
                            for pod in all_pods:
                                if pod.startswith(env_id):
                                    # save
                                    env_cfg[""]
                            print("Deployment creation awaited")
                    except Exception as e:
                        print(f"Error exec cmd: {e}")
            print(f"All resources created")
            return True
        except Exception as e:
            print(f"Unable do deploy images: {e}")
            return False


    def delelte_pods(self, pod_names:list[str]):
        # Löschbefehl für den Pod
        for pn in pod_names:
            command = ['kubectl', 'delete', 'pod', pn]
            subprocess.run(command, check=True, text=True, capture_output=True)
        print("Pod names deleted")

    def get_pod_ips(self, pod_names: list) -> dict:
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




    def get_pods(self) -> list:
        # Pods anzeigen
        print("Zeige erstellte Pods an...")

        result = subprocess.run(
            ['kubectl', 'get', 'pods'],
            check=True,
            text=True,
            capture_output=True
        )
        
        print(result.stdout)
        print("Alle Pods angezeigt.")

        pod_lines = result.stdout.strip().split('\n')
        pod_names = [line.split()[0] for line in pod_lines if line]
        return pod_names



"""
    def rm_gke_cfg_dir(self):
        self.file_store.cleanup()
        print(f"Temporary directory {self.file_store.name} removed.")ss
    def create_single_image(self, env_id, env_content): #all_pixels todo
        yaml_content = content

        manifest_file = f"{env_id}.yaml"

        path = os.path.join(
            self.file_store.name,
            manifest_file
        )

        with open(path, "w") as f:
            f.write(yaml_content)
    def get_env_section(self, env_vars):
        env_list = [f"\n                  - name: {key}\n                    value: \"{value}\"" for key, value in
                    env_vars.items()]
        env_section = f"\n                  env:{''.join(env_list)}"
        return env_section
"""

"""
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: {env_id}
              labels:
                app: {self.deployment_name}
            spec:
              replicas: 1
              selector:
                matchLabels:
                  app: {self.deployment_name}
              template:
                metadata:
                  labels:
                    app: {self.deployment_name}
                spec:
                  containers:
                  - name: {self.image_name}
                    image: {self.get_img_tag()}
                    ports:
                    - containerPort: {self.container_port}{self.get_env_section(env_vars=env_content)}
                resources: 
                  requests:
                    cpu: "4" 
                    memory: "16Gi"
                  limits:
                    cpu: "16" 
                    memory: "25Gi"           
                    
def get_deploy_commands(self, env_id: str, env_vars: dict) -> list[str]:
    print("Generating imperative kubectl commands...")                    
    # 1. Base Deployment erstellen
    
    
    
    def get_deploy_commands(self, env_id: str, env_vars: dict) -> list[list[str]]:
        print("Generating imperative kubectl commands...")
        create_depl_cmd = self.create_deployment_cmd(env_id)

        # 2. Env-Variablen hinzufügen


        # 3. Ressourcenlimits setzen
        set_res_cmd = [
            "kubectl", "set", "resources", f"deployment/{env_id}",
            "--requests=cpu=4,memory=16Gi", "--limits=cpu=16, memory=25Gi",
            "-c", self.image_name
        ]

        commands = [create_depl_cmd, set_env_cmd, set_res_cmd]
        print("Commands generated successfully.")
        return commands
    
               
"""