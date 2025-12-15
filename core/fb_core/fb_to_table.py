import os
from typing import Dict, List, Any

from utils.file._csv import collect_keys, dict_2_csv_buffer


class TimeSeriesFBConverter:
    """

    Converts Firebase data (format: key: {index: {field: value, ...}})
    into a dictionary of separate tables (List[Dict]) for each key.
    Uses instant CSV conversion helpers for export.

    """

    def __init__(self, output_dir):
        self.output_dir=output_dir # or tempdir when sending to db

    def convert_to_tables(
            self,
            entries,
            file_name,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        converts data into a dictionary of flat, row-indexed tables.
        """
        fieldnames = collect_keys(entries)
        output_dir = os.path.join(self.output_dir, file_name)
        dict_2_csv_buffer(entries, fieldnames, output_dir)
        print("saved ts data in", output_dir)




