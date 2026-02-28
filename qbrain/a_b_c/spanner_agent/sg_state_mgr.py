from qbrain.a_b_c.spanner_agent._spanner_graph.graph_loader import SpannerCore


class SpannerStateManager(SpannerCore):
    """Manages and reports resource usage for Google Cloud Spanner."""

    def __init__(self):
        SpannerCore.__init__(self)
        # Initialize client, which handles gRPC communication

    def _get_instance_resources(self, instance_id: str):
        """Gets database and session count for a specific instance."""
        instance = self.client.instance(instance_id)

        # 1. Count Databases
        databases = list(instance.list_databases())
        db_count = len(databases)

        # 2. Count Active Sessions (used for connection/query resources)
        # Note: Spanner doesn't expose memory per DB; nodes/processing units are billed.
        # Getting sessions requires a client library method call.
        session_count = 0
        try:
            # This is a client-side approximation/list of open sessions
            # The exact number of 'active' sessions is an internal metric
            session_count = len(list(instance.list_sessions()))
        except Exception:
            # Ignore error if instance is inaccessible or not ready
            pass

        return {
            "databases": db_count,
            "sessions": session_count,
            "nodes": instance.node_count,
            "processing_units": instance.processing_units
        }

    def print_resource_report(self):
        """Prints the full Spanner resource report."""
        print("-" * 50)
        print(f"📊 Spanner Resource Report for Project: {self.project_id}")

        try:
            instances = list(self.client.list_instances())
        except Exception as e:
            print(f"ERROR: Could not list Spanner instances. Check permissions/ID. Details: {e}")
            return

        total_instances = len(instances)
        print(f"\nTotal Spanner Instances: {total_instances}")

        if total_instances == 0:
            print("No Spanner resources found.")
            return

        for instance in instances:
            instance_metrics = self._get_instance_resources(instance.instance_id)
            print(f"\n  Instance: {instance.instance_id}")
            print(f"    - DBs (Amount): {instance_metrics['databases']}")
            print(f"    - Sessions (Amount): {instance_metrics['sessions']}")
            print(
                f"    - Provisioned Resources (Nodes/PUs): {instance_metrics['nodes']} Nodes / {instance_metrics['processing_units']} PUs")

        print("-" * 50)

