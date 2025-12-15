from a_b_c import BQCore

class BQStateManager(BQCore):
    """Manages and reports resource usage for Google BigQuery."""

    def __init__(self):
        BQCore.__init__(self)

    def print_resource_report(self):
        """Prints the full BigQuery resource report, including storage."""
        print("-" * 50)
        print(f"📚 BigQuery Resource Report for Project: {self.project_id}")

        try:
            datasets = list(self.bqclient.list_datasets())
        except Exception as e:
            print(f"ERROR: Could not list BigQuery datasets. Check permissions/ID. Details: {e}")
            return

        total_datasets = len(datasets)
        total_tables = 0
        total_bytes_used_gb = 0.0

        print(f"\nTotal BigQuery Datasets: {total_datasets}")

        for dataset in datasets:
            dataset_id = dataset.dataset_id

            # Count tables in the dataset
            tables = list(self.bqclient.list_tables(dataset_id))
            table_count = len(tables)
            total_tables += table_count

            # Calculate total storage size for the tables
            dataset_bytes = 0
            for table in tables:
                full_table_id = f"{dataset_id}.{table.table_id}"
                try:
                    table_ref = self.bqclient.get_table(full_table_id)
                    # Use table_ref.num_bytes as the main storage metric
                    dataset_bytes += table_ref.num_bytes
                except Exception:
                    # Table details inaccessible
                    pass

            dataset_bytes_gb = dataset_bytes / (1024 ** 3)
            total_bytes_used_gb += dataset_bytes_gb

            print(f"\n  Dataset: {dataset_id}")
            print(f"    - Tables (Amount): {table_count}")
            print(f"    - Used Storage (Estimated): {dataset_bytes_gb:.4f} GB")

        print(f"\n--- Summary ---")
        print(f"Total Tables (Amount): {total_tables}")
        print(f"Total Used Storage (Estimated): {total_bytes_used_gb:.4f} GB")
        print("-" * 50)

