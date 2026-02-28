from google.cloud import compute_v1
import os

def list_all_vm_names(project_id):
    client = compute_v1.InstancesClient()
    names = []
    request = compute_v1.AggregatedListInstancesRequest(project=project_id)
    for zone, response in client.aggregated_list(request=request):
        if response.instances:
            for instance in response.instances:
                names.append(instance.name)

    # Print all available m names
    for name in names:
        print(name)

    return names

if __name__ == '__main__':
    project_id = os.environ.get('GCP_PROJECT_ID', "aixr-401704")
    if not project_id:
        print("Bitte setze die Umgebungsvariable 'GCP_PROJECT_ID'.")
    else:
        vm_names = list_all_vm_names(project_id)
        for name in vm_names:
            print(name)