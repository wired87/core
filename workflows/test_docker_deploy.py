"""
Test for simplified Docker deployment.
Runs build/tag/push via exec_cmd within the test.

Commands to run the test (from project root):
  py -m unittest workflows.test_docker_deploy -v
  py -m unittest workflows.test_docker_deploy.TestDockerDeploy.test_deploy_simple_build_only -v
  py -m workflows.test_docker_deploy

Requires: Docker Desktop running, GCP auth (set_gcp_auth_path).
Tests are skipped when Docker is not available.
"""
import os
import subprocess
import unittest

from auth.set_gcp_auth_creds_path import set_gcp_auth_path
from utils.run_subprocess import exec_cmd
from workflows.docker_deploy import (
    docker_build,
    docker_tag,
    docker_run_local,
    deploy_simple,
)


def _docker_available():
    """Check if Docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


DOCKER_AVAILABLE = _docker_available()


@unittest.skipIf(not DOCKER_AVAILABLE, "Docker Desktop not running or not installed")
class TestDockerDeploy(unittest.TestCase):
    """Test simplified docker deployment via exec_cmd."""

    @classmethod
    def setUpClass(cls):
        set_gcp_auth_path()

    @property
    def project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_docker_build(self):
        """Build image via exec_cmd."""
        result = docker_build(
            image_name="qfs-test",
            dockerfile_path=os.path.join(self.project_root, "Dockerfile"),
            context=self.project_root,
        )
        self.assertEqual(result, "qfs-test")

    def test_docker_tag(self):
        """Tag local image via exec_cmd."""
        result = docker_tag(
            "qfs-test",
            "us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs-test:latest",
        )
        self.assertIsNotNone(result)

    def test_docker_run_local(self):
        """Run container locally via exec_cmd, then stop."""
        out = docker_run_local(
            image="qfs-test",
            name="qfs-test-run",
            env={"START_MODE": "TEST"},
            detach=True,
        )
        self.assertIsNotNone(out)
        exec_cmd(["docker", "rm", "-f", "qfs-test-run"])

    def test_deploy_simple_build_only(self):
        """Deploy simple: build only (no push) via exec_cmd."""
        uri = deploy_simple(
            image_name="qfs-test",
            context=self.project_root,
            build_only=True,
        )
        self.assertIn("qfs-test", uri)
        self.assertIn("us-central1-docker.pkg.dev", uri)


if __name__ == "__main__":
    unittest.main()
