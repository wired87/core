import os
import subprocess

from pathlib import Path
import docker

from dockerfile import get_custom_dockerfile_content
from dynamic_docker import generate_dockerfile

class DockerAdmin:
    def __init__(self, context_path: str = ".", dockerfile_name: str = "Dockerfile"):
        self.context_path = Path(context_path)
        self.dockerfile_path = self.context_path / dockerfile_name
        self.client = docker.from_env()

    def build_docker_image(self, image_name, dockerfile_path='.', e={}):
        env_str = " ".join([f'--env {name}="{val}"' for name, val in e.items()])
        try:
            command = f"docker build -t {image_name} {dockerfile_path} -e {env_str}"
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Image '{image_name}' successfully built.")
        except subprocess.CalledProcessError as e:
            print(f"Error building image: {e.stderr}")
        except FileNotFoundError:
            print("Docker is not installed or not in the system's PATH.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def run_local_docker_image(
            self,
            image: str,
            name: str = None,
            ports: dict = None,
            env: dict = None,
            detach: bool = True
    ) -> str:
        """
        Run a local Docker image for testing.

        :param image: Docker image name (e.g. "my-app:latest")
        :param name: Optional container name
        :param ports: Dict of {host_port: container_port}, e.g. {8000: 8000}
        :param env: Dict of environment variables {key: value}
        :param detach: Run in background (default: True)
        :return: Container ID
        """
        try:
            cmd = ["docker", "run"]

            if detach:
                cmd.append("-d")

            if name:
                cmd += ["--name", name]

            if ports:
                for host_port, container_port in ports.items():
                    cmd += ["-p", f"{host_port}:{container_port}"]

            if env:
                for k, v in env.items():
                    cmd += ["-e", f"{k}={v}"]

            cmd.append(image)

            print("Running local docker container:", " ".join(cmd))
            container_id = subprocess.check_output(cmd, text=True).strip()
            print(f"✅ Started container {container_id[:12]} from {image}")
            return container_id
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run docker image: {e}")
            return None

    # --- Dockerfile-Erstellung ---
    def create_static_dockerfile(
        self,
        base_ray_image: str,
        requirements_file: str,
        app_script_name: str
    ):
        """Erstellt ein statisches Dockerfile aus Basis-Template."""
        content = get_custom_dockerfile_content(
            base_ray_image=base_ray_image,
            requirements_file=requirements_file,
            app_script_name=app_script_name
        )
        self._write_dockerfile(content)
        print(f"✅ Static Dockerfile erstellt: {self.dockerfile_path}")

    def create_dynamic_dockerfile(self, project_root, startup_cmd, **env_vars):
        """Erstellt ein dynamisches Dockerfile mit ENV-Variablen."""
        content = generate_dockerfile(
            project_root=project_root,
            startup_cmd=startup_cmd,
            **env_vars
        )
        print(f"✅ Dynamic Dockerfile erstellt: {self.dockerfile_path}")

        return content



    def _write_dockerfile(self, content: str):
        self.dockerfile_path.write_text(content)



    # --- Docker Image Build ---
    def build_image(self, path, image_name: str, tag: str = "latest"):
        """Baut ein Image, wenn es nicht bereits existiert."""
        """
        Prüft, ob ein Docker-Image existiert. Wenn nicht, wird es gebaut.

        Args:
            path (str): Lokaler Pfad zum Docker-Kontext (enthält Dockerfile)
            image_name (str): Name des Docker-Images
            tag (str): Tag (default: "latest")
        """
        client = docker.from_env()

        # Check if image exists
        images = client.images.list(name=image_name)
        for img in images:
            if f"{image_name}:{tag}" in img.tags:
                print(f"✅ Image '{image_name}:{tag}' exists.")
                return

        print(f"🔨 Baue Image: {image_name}:{tag}")
        try:
            image, logs = client.images.build(path=path, tag=f"{image_name}:{tag}")
            for line in logs:
                if 'stream' in line:
                    print(line['stream'].strip())
            print(f"✅ Build erfolgreich: {image_name}:{tag}")
        except Exception as e:
            print(f"❌ Fehler beim Build: {e}")

    def force_build_image(self, image_name: str, tag: str = "latest"):
        """Baut ein Image immer neu."""
        print(f"🔨 Force-Build: {image_name}:{tag}")
        try:
            image, logs = self.client.images.build(
                path=str(self.context_path),
                tag=f"{image_name}:{tag}"
            )
            for line in logs:
                if 'stream' in line:
                    print(line['stream'].strip())
            print(f"✅ Build erfolgreich: {image_name}:{tag}")
        except Exception as e:
            print(f"❌ Fehler beim Build: {e}")

    # --- Utility ---
    def image_exists(self, image_name: str, tag: str = "latest") -> bool:
        """Prüft, ob ein Image mit Tag existiert."""
        images = self.client.images.list(name=image_name)
        return any(f"{image_name}:{tag}" in img.tags for img in images)

vars_dict = {
    "DOMAIN": os.environ.get("DOMAIN"),
    "USER_ID": os.environ.get("USER_ID"),
    "GCP_ID": os.environ.get("GCP_ID"),
    "ENV_ID": os.environ.get("ENV_ID"),
    "INSTANCE": os.environ.get("FIREBASE_RTDB"),
    "STIM_STRENGTH": os.environ.get("STIM_STRENGTH"),
}
r"""
# --- Beispielnutzung ---
if __name__ == "__main__":
    admin = DockerAdmin(".")
    ray_admin = RayAdminBase()

    for sub in ALL_SUBS:
        vars_dict.update({"FIELD_TYPE": sub})
        content = admin.create_dynamic_dockerfile(
            roject_root=rf"C:\Users\wired\OneDrive\Desktop\Projects\qfs",
            startup_cmd="python gpu_gauge_arsenal.py",
            **vars_dict
        )

        path = os.path.join(fr"C:\Users\wired\OneDrive\Desktop\qfs\docker","bb_{sub}"), "v1")
        admin.build_docker_image(path, dockerfile_path=current_dir)
"""