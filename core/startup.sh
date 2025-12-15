#!/bin/bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# Enable and start the Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Configure gcloud credentials for docker login to Artifact Registry
# This uses the VM's service account credentials
gcloud auth configure-docker {self.region}-docker.pkg.dev --quiet

# Pull the image from Artifact Registry
sudo docker pull {self.image_uri}

# Run the container in detached mode, mapping the container port
sudo docker run -d -p {self.port}:{self.port} {self.image_uri}