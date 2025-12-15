







# 2 cpu, 4gib ram

create_=r"""
gcloud compute instances create betse2 --project=aixr-401704 --zone=us-central1-c --machine-type=e2-standard-4 --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default --metadata=enable-osconfig=TRUE --maintenance-policy=MIGRATE --provisioning-model=STANDARD --service-account=aixr-401704@appspot.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append --tags=http-server,https-server,lb-health-check --create-disk=auto-delete=yes,boot=yes,device-name=betse2,disk-resource-policy=projects/aixr-401704/regions/us-central1/resourcePolicies/default-schedule-1,image=projects/debian-cloud/global/images/debian-12-bookworm-v20250415,mode=rw,size=30,type=pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud --reservation-affinity=any && printf 'agentsRule:\n  packageState: installed\n  version: latest\ninstanceFilter:\n  inclusionLabels:\n  - labels:\n      goog-ops-agent-policy: v2-x86-template-1-4-0\n' > config.yaml && gcloud compute instances ops-agents policies create goog-ops-agent-v2-x86-template-1-4-0-us-central1-c --project=aixr-401704 --zone=us-central1-c --file=config.yaml
"""

def init_debian(repo):
    return rf"""
sudo apt update && sudo apt install python3.11-venv python3-tk git tmux -y && \
python3 -m venv workenv && git clone https://github.com/wired87/qfs.git && \
export PYTHONPATH=$PYTHONPATH:$(pwd) && source workenv/bin/activate && cd qfs && pip install -r r.txt  

DJ + && \ python manage.py migrate && python manage.py collectstatic 
"""

ENDODE_META=f"""
python3 data/extractors/functions/encode/cell_line_processor/metadata_processor.py
"""


PULL_SUB = "cd qf_sim && git pull && cd .."

#sudo tail -n 100 /var/log/nginx/error.log

# 18728
RUN_ENCODE = "data/extractors/functions/encode/cell_line_processor/metadata_processor.py"
RUN_THALMUS_TRANSCRIPT = "export PYTHONPATH=$PYTHONPATH:$(pwd) gnn/processing/layer/cell_layer.py"
R_CELL_LIKNE_PROC=rf"""python3 data/extractors/functions/encode/cell_line_processor/cell_line_processor.py"""
REMBEDER = f"""python3 ggoogle/spanner/dj/views/embedder.py"""

IMIT=rf"""
source workenv/bin/activate && cd qfs && export PYTHONPATH=$PYTHONPATH:$(pwd)
"""
token="REDACTED_GITHUB_TOKEN"
RUN_PROTEIN=f"gnn/processing/layer/uniprot/main.py"
run = "python3 gnn/processing/layer/gene_layer.py"

check_fiel_size="""stat --format="%s bytes" train_data/non_neuronal_cell.h5ad"""
restart = """
cd work && source workenv/bin/activate && cd brainmaster_processing && export PYTHONPATH=$PYTHONPATH:$(pwd)  
"""
