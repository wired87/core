from django.views import View
from django.http import JsonResponse
import subprocess

from gke.build_admin import GKEAdmin


# Importiere deine GKEAdmin-Klasse


class DeletePodView(View):
    """
    Ein View, der einen GKE-Pod mit einem bestimmten Namen löscht.
    """

    def delete(self, request, *args, **kwargs):
        pod_names: list = kwargs.get('pod_names')
        cluster_name = kwargs.get('cluster_name', "autopilot-cluster-1")

        if not pod_names or not isinstance(pod_names, list) or not len(pod_names):
            return JsonResponse({'error': 'Pod name not provided'}, status=400)

        admin = GKEAdmin()

        try:
            # Authentifizierung beim Cluster
            admin.authenticate_cluster(
                cluster_name
            )
            admin.delelte_pods(pod_names)

            return JsonResponse({'message': f'Pods successfully deleted'}, status=200)

        except subprocess.CalledProcessError as e:
            # Fehler behandeln, wenn der Pod nicht existiert oder ein anderer Fehler auftritt
            return JsonResponse({'error': f'Failed to delete pod: {e.stderr.strip()}'}, status=500)
        except Exception as e:
            # Unerwartete Fehler
            return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)