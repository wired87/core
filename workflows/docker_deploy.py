"""
Simplified Docker deployment using exec_cmd.
Build, tag, push, and run locally via subprocess.
"""
import os
from typing import Optional

from utils.run_subprocess import exec_cmd, pop_cmd


def docker_build(
    image_name: str = "qfs",
    dockerfile_path: str = ".",
    context: str = None,
) -> Optional[str]:
    """Build Docker image via exec_cmd. Returns stdout or None on failure."""
    if context is None:
        context = os.path.dirname(dockerfile_path) if os.path.isfile(dockerfile_path) else (dockerfile_path or ".")
    df = dockerfile_path if os.path.isfile(dockerfile_path) else os.path.join(context, "Dockerfile")
    cmd = ["docker", "build", "-t", image_name, "-f", df, context]
    print(f"[docker_deploy] build: {' '.join(cmd)}")
    pop_cmd(cmd)  # stream output for long builds
    return image_name


def docker_tag(local_image: str, remote_uri: str) -> Optional[str]:
    """Tag local image for Artifact Registry."""
    cmd = ["docker", "tag", local_image, remote_uri]
    print(f"[docker_deploy] tag: {' '.join(cmd)}")
    return exec_cmd(cmd)


def docker_push(remote_uri: str) -> Optional[str]:
    """Push image to Artifact Registry."""
    cmd = ["docker", "push", remote_uri]
    print(f"[docker_deploy] push: {' '.join(cmd)}")
    pop_cmd(cmd)
    return remote_uri


def docker_run_local(
    image: str,
    name: Optional[str] = None,
    env: Optional[dict] = None,
    detach: bool = True,
) -> Optional[str]:
    """Run container locally for testing."""
    cmd = ["docker", "run"]
    if detach:
        cmd.append("-d")
    if name:
        cmd.extend(["--name", name])
    if env:
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
    cmd.append(image)
    print(f"[docker_deploy] run: {' '.join(cmd)}")
    out = exec_cmd(cmd)
    return out


def deploy_simple(
    image_name: str = "qfs",
    project_id: str = None,
    region: str = "us-central1",
    repo: str = "qfs-repo",
    tag: str = "latest",
    dockerfile_path: str = None,
    context: str = ".",
    build_only: bool = False,
) -> str:
    """
    Simplified deploy: build -> tag -> push.
    Uses exec_cmd for all steps.
    """
    project_id = project_id or os.environ.get("GCP_PROJECT_ID", "aixr-401704")
    remote_uri = f"{region}-docker.pkg.dev/{project_id}/{repo}/{image_name}:{tag}"
    df = dockerfile_path or os.path.join(context, "Dockerfile")

    docker_build(image_name=image_name, dockerfile_path=df, context=context)
    docker_tag(image_name, remote_uri)
    if not build_only:
        docker_push(remote_uri)
    return remote_uri
