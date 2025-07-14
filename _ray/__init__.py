"""

https://console.anyscale.com/cloud-setup/customer-hosted

pip install -U "anyscale[gcp]"
anyscale login # authenticate



# Setup VM -> GKE communication
# sudo apt-get install kubectl




Struktur:
Erstlle immer ein Dockerfile für das gesammte _qfn_cluster_node



Start
ray start --head --port=6379



















from _google.gke._ray.qf_core_base import RayBase

_ray = RayBase()
_ray.start()

"""
import ray

print(ray.is_initialized())
