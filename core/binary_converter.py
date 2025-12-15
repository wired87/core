import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Define the expected main file name
MAIN_SCRIPT_NAME = "main.py"
# Define the output executable name
OUTPUT_EXECUTABLE_NAME = "HostableApp"


class CompilerEngine:
    """
    Engine for scanning a directory and compiling a Python project
    into a single, hostable binary using Nuitka.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.main_file_path: Optional[Path] = None
        self.output_dir = self.root_dir / "build_output"
        self.output_json_path = self.root_dir / "compilation_report.json"

    def _scan_directory(self) -> bool:
        """Walks local directory and finds the main script."""
        print(f"🔍 Scanning directory: {self.root_dir}")

        # Check if main.py exists in the root
        main_path = self.root_dir / MAIN_SCRIPT_NAME
        if main_path.exists():
            self.main_file_path = main_path
            print(f"✅ Found main script: {self.main_file_path}")
            return True

        print(f"❌ '{MAIN_SCRIPT_NAME}' not found in the root directory.")
        return False

    def _compile_to_binary(self) -> Dict[str, Any]:
        """Executes Nuitka to compile the main script into a single executable."""
        if not self.main_file_path:
            return {"status": "FAILED", "error": "No main script found."}

        # Ensure the output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # --- Nuitka Command ---
        # --standalone: Creates a folder with all dependencies (DLLs/Libs)
        # --onefile: Creates a single executable (requires standalone)
        # --include-directory: Includes the entire source directory

        nuitka_command = [
            "python", "-m", "nuitka",
            "--standalone",
            "--onefile",
            f"--output-dir={self.output_dir}",
            f"--output-filename={OUTPUT_EXECUTABLE_NAME}",
            f"--include-data-dir={self.root_dir}=.",  # Include the whole source tree as data (if needed)
            str(self.main_file_path)
        ]

        print("🔨 Starting Nuitka compilation...")
        try:
            # Execute the command
            result = subprocess.run(
                nuitka_command,
                capture_output=True,
                text=True,
                check=True
            )

            # The executable location depends on the OS (Windows: .exe, Linux: no extension)
            executable_name = OUTPUT_EXECUTABLE_NAME + (".exe" if os.name == 'nt' else "")
            final_path = self.output_dir / executable_name

            return {
                "status": "SUCCESS",
                "binary_path": str(final_path.resolve()),
                "output_dir": str(self.output_dir.resolve()),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.CalledProcessError as e:
            print(f"❌ Nuitka Error: {e.stderr}")
            return {
                "status": "FAILED",
                "error": "Compilation failed (Nuitka or C++ compiler issue).",
                "details": e.stderr
            }
        except FileNotFoundError:
            return {"status": "FAILED", "error": "Python or Nuitka not found. Ensure they are in PATH."}

    def run_compilation(self) -> str:
        """Main execution workflow."""
        if not self._scan_directory():
            return "❌ Compilation failed: Main script not found."

        report = self._compile_to_binary()

        # Save the report structure (as requested output)
        with open(self.output_json_path, 'w') as f:
            json.dump(report, f, indent=2)

        if report['status'] == 'SUCCESS':
            print(f"🥳 Compilation successful! Hostable file saved to: {report['binary_path']}")
            print(f"💾 Report saved to: {self.output_json_path}")
            return report['binary_path']
        else:
            print(f"🛑 Error Report saved to: {self.output_json_path}")
            return f"🛑 Compilation failed. Check {self.output_json_path} for details."


# --- Workflow Testing ---

def setup_test_project(root: Path):
    """Creates a minimal test file structure."""
    if not root.exists():
        root.mkdir()

    # Create the main script
    (root / MAIN_SCRIPT_NAME).write_text(
        """
        import sys
        import os
        
        print(f"Hello, I am a compiled binary from Python {sys.version.split(' ')[0]}")
        print(f"Running in directory: {os.getcwd()}")
        """
    )
    print(f"✨ Test project created at {root}")


def test_cli_action():
    """Defines and executes the compilation workflow as a testable CLI action."""
    test_dir = Path("compilation_test_project")
    setup_test_project(test_dir)

    engine = CompilerEngine(str(test_dir))
    final_binary_path = engine.run_compilation()

    print(f"\nFinal Result Path: {final_binary_path}")

    # Cleanup logic (optional, but good practice)
    # import shutil
    # if test_dir.exists():
    #     shutil.rmtree(test_dir)

    print("\n--- ✅ Compilation Test Workflow Finished ---")


if __name__ == '__main__':
    test_cli_action()