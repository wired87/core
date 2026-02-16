#!/bin/bash
        set -e
    
        echo "--- Starte System-Vorbereitung ---"
        
        # 1. Docker installieren (falls nicht vorhanden)
        if ! command -v docker &> /dev/null; then
            apt-get update
            apt-get install -y docker.io
            systemctl start docker
            systemctl enable docker
        fi
    
        # 2. NVIDIA Container Toolkit installieren
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update
            apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker
        fi
    
        # 3. Authentifizierung an der Google Artifact Registry
        gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
    
        # 4. Container ziehen und starten
        echo "--- Starte Container: qfs@sha256 ---"
        docker pull us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs@sha256:355b3ba46adf6456d27c66ab234e8650cc4997df9911ca8957bdb0f09d21f5c0
    
        # Falls ein alter Container mit gleichem Namen läuft, diesen entfernen
        docker rm -f qfs@sha256 || true
    
        docker run -d \
            --name qfs@sha256 \
            --restart always \
            --gpus all \
            -e DOMAIN='www.bestbrain.tech' -e GCP_ID='aixr-401704' -e DATASET_ID='QBRAIN' -e BQ_DATA_TABLE='env_7c87bb26138a427eb93cab27d0f5429f_data' -e ENV_ID='env_7c87bb26138a427eb93cab27d0f5429f' -e USER_ID='72b74d5214564004a3a86f441a4a112f' -e DELETE_POD_ENDPOINT='gke/delete-pod/' -e SG_DB_ID='env_7c87bb26138a427eb93cab27d0f5429f' -e BQ_PROJECT='aixr-401704' -e BQ_DATASET='QBRAIN' -e START_TIME='300' -e AMOUNT_NODES='64' -e DIMS='3' \
            us-central1-docker.pkg.dev/aixr-401704/qfs-repo/qfs@sha256:355b3ba46adf6456d27c66ab234e8650cc4997df9911ca8957bdb0f09d21f5c0
    
        echo "--- Startup script finished - Container is running ---"
        