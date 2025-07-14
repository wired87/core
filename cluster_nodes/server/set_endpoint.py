

def set_endpoint(env_id):
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    address = f"ray://{host_ip}:10001"
    endpoint = address + f"/{env_id}"
    return endpoint