import json

from qbrain.core.app_utils import USER_ID, ENV_ID


class LogAIExplain:
    """Analysiert Logdaten mit einem KI-Modell und erstellt eine Zusammenfassung."""

    def __init__(
            self,
            db_manager,
            env_id=ENV_ID,
            user_id=USER_ID,
    ):
        """Initialisiert die Klasse mit einer DBManager-Instanz."""
        self.db_manager = db_manager

        self.user_id = user_id
        self.env_id = env_id

    def analyze_logs(self, nid):
        """Ruft Logdaten ab und erstellt eine Zusammenfassung."""
        err_logs = self.db_manager.get_latest_entries(
            path=f"users/{self.user_id}/env/{self.env_id}/logs/{nid}/err/"
        )

        out_logs = self.db_manager.get_latest_entries(
            path=f"users/{self.user_id}/env/{self.env_id}/logs/{nid}/out/"
        )

        prompt = (
            "Analyze the following logs and provide a concise summary of the events and processes. "
            "Identify any anomalies or notable occurrences.\n\n"
            f"Error Logs:\n{json.dumps(err_logs, indent=2)}"
            f"==========================="
            f"Output Logs:\n{json.dumps(out_logs, indent=2)}"
        )
        response = None#ask_gem(prompt)
        return response

