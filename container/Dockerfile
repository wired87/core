# Use a slim Python image
FROM python:3.10-slim AS base

# Set build argument for GitHub Access Key (Passed at build time)

# Set a dedicated working directory (not root)
WORKDIR /usr/src/app

### 🔹 SUBMODULE HANDLING (Using HTTPS with Token)

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/cluster_nodes

### 🔹 MAIN BUILD

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential  \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching dependencies
COPY qf_sim/clusters/manager ./cluster_nodes
COPY qf_sim/clusters/cluster_utils ./cluster_nodes
COPY qf_sim/clusters/server ./cluster_nodes
COPY utils .

# Install dependencies with no cache
RUN pip install -r r.txt

# Copy the rest of the project files into the working directory
COPY head .

# Set environment variables
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DJANGO_SETTINGS_MODULE=bm.settings \
    #DEBUG=0 \
    GCS_MOUNT_PATH=/mnt/bucket/
    DOMAIN=bestbrain.tech \
    CREDS_REQUEST_ENDPOINT=/auth/gh_access



# Expose the application port
EXPOSE 8080

# Run Django using Gunicorn
CMD ["./server/main.py"]