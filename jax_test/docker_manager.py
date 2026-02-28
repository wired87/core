import subprocess
import time
import sys

class DockerTestRunner:
    """
    A test class utility to build, run, and debug the JAX GNN Docker container.
    """
    
    def __init__(self, image_name="jax_gnn_test", container_name="jax_gnn_container"):
        self.image_name = image_name
        self.container_name = container_name

    def run_command(self, command):
        """Helper to run shell commands and stream output."""
        print(f"Executing: {command}")
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if stdout:
            print(stdout)
        if stderr:
            print(f"Error/Warning:\n{stderr}")
            
        return process.returncode == 0

    def build(self):
        """Builds the Docker image."""
        print(f"--- Building Docker Image: {self.image_name} ---")
        return self.run_command(f"docker build -t {self.image_name} .")

    def run(self):
        """Runs the Docker container in a test mode."""
        print(f"--- Running Container: {self.image_name} ---")
        # Run and remove after execution
        return self.run_command(f"docker run --rm --name {self.container_name} {self.image_name}")

    def debug_shell(self):
        """Starts an interactive shell inside the container."""
        print(f"--- Starting Interactive Shell in: {self.image_name} ---")
        try:
            # We use subprocess.call here to let the user interact with the shell
            subprocess.call(f"docker run --rm -it {self.image_name} /bin/bash", shell=True)
        except KeyboardInterrupt:
            print("\nExiting debug shell.")

if __name__ == "__main__":
    runner = DockerTestRunner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        runner.build()
        runner.debug_shell()
    else:
        if runner.build():
            print("Build Successful.")
            if runner.run():
                print("Test Run Successful.")
            else:
                print("Test Run Failed.")
        else:
            print("Build Failed.")
