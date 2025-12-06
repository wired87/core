import os
import subprocess
import sys

import docker

from bob_builder._docker.dockerfile import get_custom_dockerfile_content
from bob_builder._docker.dynamic_docker import generate_dockerfile
from utils.run_subprocess import pop_cmd


class DockerAdmin:
    # pip install "docker==6.1.3"

    def __init__(self):
        self.client = docker.api.client.APIClient()

    def login_to_artifact_registry(self, region: str):
        """Authenticates Docker with Google Artifact Registry."""
        print(f"Configuring Docker for Artifact Registry in region: {region}")
        try:
            cmd = ["gcloud", "auth", "configure-docker", f"{region}-docker.pkg.dev", "--quiet"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Docker authenticated with Artifact Registry successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to authenticate with Artifact Registry: {e.stderr}")
            raise
        except FileNotFoundError:
            print("❌ 'gcloud' command not found. Is the Google Cloud SDK installed and in your PATH?")
            raise


    def build_docker_image(self, image_name, dockerfile_path='.', e=None):
        cmd = f"docker build -t {image_name} {dockerfile_path}"
        if e is not None:
            env_str = " ".join([f'--env {name}="{val}"' for name, val in e.items()])
            cmd +=  f"-e {env_str}"
        pop_cmd(cmd)


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
        """
        try:
            cmd = ["docker", "run"]
            if detach: cmd.append("-d")
            if name: cmd += ["--name", name]
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

    def create_static_dockerfile(
        self,
        base_ray_image: str,
        requirements_file: str,
        app_script_name: str
    ):
        content = get_custom_dockerfile_content(
            base_ray_image=base_ray_image,
            requirements_file=requirements_file,
            app_script_name=app_script_name
        )
        self._write_dockerfile(content)
        print(f"✅ Static Dockerfile erstellt: {self.dockerfile_path}")

    def create_dynamic_dockerfile(self, project_root, startup_cmd, **env_vars):
        content = generate_dockerfile(
            project_root=project_root,
            startup_cmd=startup_cmd,
            **env_vars
        )
        print(f"✅ Dynamic Dockerfile erstellt: {self.dockerfile_path}")
        return content

    def build_image(self, path, image_name: str):
        """
            Builds a Docker image using the given tag and context path.

            Args:
                tag: The tag for the Docker image (e.g., 'qfs').
                context_path: The build context path (e.g., '.').
            """
        command = ["docker", "build", "-t", image_name]
        print(f"Running command: {' '.join(command)}")
        try:
            # Use subprocess.run to execute the command.
            # check=True raises a CalledProcessError if the command returns a non-zero exit code.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Stream the output in real-time
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')

            process.wait()

            if process.returncode != 0:
                print(f"Error: Docker build failed with exit code {process.returncode}", file=sys.stderr)
                sys.exit(1)

            print("\nDocker image built successfully!")

        except FileNotFoundError:
            print("Error: 'docker' command not found. Is Docker installed and in your PATH?", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)

    def force_build_image(self, image_name: str, tag: str = "latest"):
        """Baut ein Image immer neu."""
        print(f"🔨 Force-building: {image_name}:{tag}")
        try:
            image, logs = self.client.images.build(
                path=str(self.context_path),
                tag=f"{image_name}:{tag}"
            )
            for line in logs:
                if 'stream' in line:
                    print(line['stream'].strip())
            print(f"✅ Build successful: {image_name}:{tag}")
        except Exception as e:
            print(f"❌ Build error: {e}")

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
