import subprocess
import sys

def build_image(image_name: str, tag: str = "latest", dockerfile: str = "Dockerfile", context: str = "."):

    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile not found at: {dockerfile}")

    cmd = [
        "docker", "build",
        "-t", f"{image_name}:{tag}",
        "-f", str(dockerfile),
        str(context)
    ]

    print(f"🔧 Building image {image_name}:{tag} ...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"✅ Build successful: {image_name}:{tag}")
    else:
        print("❌ Build failed:")
        print(result.stderr)
        sys.exit(result.returncode)


