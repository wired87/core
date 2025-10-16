import os.path


headers = {
    "Content-Type": "application/json"   # <-- MANDATORY
}

if __name__ == "__main__":
    print(os.path.relpath("AUTHORS.md", start="_betse"))