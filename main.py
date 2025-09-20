import os.path


headers = {
    "Content-Type": "application/json"   # <-- MANDATORY
}

if __name__ == "__main__":
    r"""content=load_yaml(r"C:\Users\wired\OneDrive\Desktop\base_dj\_betse\betse-1.5.0\betse\data\yaml\sim_config.yaml")
    pprint.pp(content)
    r = requests.post("http://127.0.0.1:8000/betse/run/", json={"sim_config_data": content }, headers=headers)
    print(r.json())"""
    print(os.path.relpath("AUTHORS.md", start="_betse"))