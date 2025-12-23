from a_b_c.spanner_agent import GCP_ID
from utils.run_subprocess import exec_cmd
"""
Update gcloud cli version > 501 (as admin)
"""

class SpannerEmulatorManager:
    """Manages the Spanner Emulator lifecycle and configuration via gcloud CLI."""
    def __init__(self, project_id=None):
        self.project_id = project_id or GCP_ID


    def get_emu(self) -> str:
        """Installs Spanner emulator component and updates all gcloud components."""
        exec_cmd(["gcloud", "components", "install", "cloud-spanner-emulator"])

    # Assuming exec_cmd is defined elsewhere (e.g., in self or imported)

    def create_and_configure_emulator_config(self) -> str:
        """
        Creates and activates the 'emulator' config, disables auth, and sets endpoint.
        All gcloud commands are collected and executed in sequence.
        """

        # 1. Collect all commands into a single list
        commands: list[list[str]] = [
            # Create and activate config
            ["gcloud", "config", "configurations", "create", "emulator"],
            ["gcloud", "config", "configurations", "activate", "emulator"],

            # Set auth and project
            ["gcloud", "config", "set", "auth/disable_credentials", "true"],
            #["gcloud", "config", "set", "project", self.project_id],

            # Set endpoint override (Final command whose output is returned)
            ["gcloud", "config", "set", "api_endpoint_overrides/spanner", "http://localhost:9020/"]
        ]

        last_output: str = ""

        # 2. Loop and execute commands with error handling
        try:
            for cmd in commands:
                print("Exec:", cmd)
                # Assuming exec_cmd handles the command execution and returns its output
                last_output = exec_cmd(cmd)  # Using self.exec_cmd based on class context

        except Exception as e:
            # Minimalist error handling: raise a clear exception indicating failure
            print(f"Failed to execute gcloud command: {' '.join(cmd)}. Error: {e}")

        # 3. Return the output of the last successfully executed command
        return last_output


    def main(self):
        print("=========== START SPANNER EMULATOR ===========")
        self.get_emu()
        self.create_and_configure_emulator_config()



if __name__ == "__main__":
    sem=SpannerEmulatorManager()
    sem.main()