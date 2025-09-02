def create_cluster():
    return """
gcloud beta container --project "aixr-401704" clusters create-auto "autopilot-cluster-1" --region "us-central1" --release-channel "regular" --enable-dns-access --enable-ip-access --no-enable-google-cloud-access --network "projects/aixr-401704/global/networks/default" --subnetwork "projects/aixr-401704/regions/us-central1/subnetworks/default" --cluster-ipv4-cidr "/17" --binauthz-evaluation-mode=DISABLED --fleet-project=aixr-401704 --enable-ray-operator --enable-ray-cluster-logging --enable-ray-cluster-monitoring
    """