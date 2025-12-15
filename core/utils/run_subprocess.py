import os
import subprocess

def exec_cmd(cmd, inp=None):
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            input=inp,
            shell=os.name == "nt",
            capture_output=True,
            timeout=60,
        )
        if result is not None:
            result = result.stdout.strip()
        # print("CMD result:", result)
        return result
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        print("Output:", e.stdout)

