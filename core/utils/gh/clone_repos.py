
import os

from utils.run_subprocess import exec_cmd


def clone_repos():
    repo_base = "https://github.com/wired87/"
    repo_names = [
        "qf_sim",
        "a_b_c",
        "_ray_core",
        "fb_core",
    ]

    target_dir = "subrepos"
    os.makedirs(target_dir, exist_ok=True)

    for name in repo_names:
        repo_url = f"{repo_base}{name}.git"
        clone_path = os.path.join(target_dir, name)

        if os.path.exists(clone_path):
            print(f"→ {name} already exists, pulling latest changes...")
            exec_cmd(["git", "-C", clone_path, "pull", "--rebase"])
        else:
            print(f"📦 Cloning {repo_url} ...")
            exec_cmd(["git", "clone", "--depth", "1", repo_url, clone_path])

    print("✅ All repositories cloned or updated.")