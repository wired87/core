from google.cloud import bigquery
from typing import Dict, Any, List, Optional, Tuple

from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from core.app_utils import TABLE_NAME, ENV_ID


class BQMLMaster(BQCore):
    """
    Klasse zur Kapselung des Trainings von BigQuery ML Modellen.

    Erforderliche Trainingsparameter für Zeitreihen-Modelle (ARIMA_PLUS_XREG):
    1. model_name (Modellziel)
    2. training_table (Quell-Tabelle)
    3. time_series_timestamp_col (TIME_SERIES_TIMESTAMP_COL)
    4. time_series_data_col (TIME_SERIES_DATA_COL)
    5. time_series_id_col (TIME_SERIES_ID_COL)
    """

    def __init__(self, dataset_id=None):
        BQCore.__init__(self, dataset_id)

    def _format_bqml_options(self, options: Dict[str, Any]) -> str:
        """
        Formatiert ein Dictionary von Optionen in den OPTIONS(...) SQL-String.
        """
        options_list = []
        for key, value in options.items():
            # Schlüssel in Großbuchstaben konvertieren (BQML-Konvention)
            key = key.upper()

            if isinstance(value, str) and key not in ['NON_SEASONAL_ORDER', 'KMS_KEY_NAME']:
                # Normale Strings in einfache Anführungszeichen setzen
                formatted_value = f"'{value}'"
            elif isinstance(value, (int, float, bool)):
                # Numerische und boolesche Werte direkt verwenden
                formatted_value = str(value).upper() if isinstance(value, bool) else str(value)
            elif key == 'NON_SEASONAL_ORDER':
                # Tupel p, d, q direkt als String übergeben
                formatted_value = str(value)
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                # Liste von Strings (z.B. HOLIDAY_REGION)
                formatted_value = f"[{', '.join(str(v) for v in value)}]"

            else:
                # Standard-Behandlung (z.B. für Spaltennamen)
                formatted_value = str(value)

            options_list.append(f"{key}={formatted_value}")

        return ",\n" + ",\n".join(options_list) if options_list else ""

    def train_time_series_model(
            self,
            model_name: str="hi1",  # 1. ZIEL: Modellname (z.B. `mydataset.mymodel`)
            training_table: str=TABLE_NAME,  # 2. QUELLE: Tabelle für das Training

            # --- Erforderliche BQML Parameter ---
            time_series_timestamp_col: str="tid",  # TIME_SERIES_TIMESTAMP_COL
            time_series_data_col: str=[
                "psi",
                "psi_bar",
                "field_value",
                "",
            ],  # TIME_SERIES_DATA_COL
            time_series_id_col: str="nid",  # TIME_SERIES_ID_COL

            # --- Optionale Zeitreihen / ARIMA Parameter ---
            model_type: str = 'ARIMA_PLUS_XREG',
            horizon: Optional[int] = None,
            auto_arima: bool = False,
            auto_arima_max_order: Optional[int] = None,
            auto_arima_min_order: Optional[int] = None,
            non_seasonal_order: Optional[Tuple[int, int, int]] = None,  # p, d, q Tupel
            data_frequency: Optional[str] = None,
            include_drift: bool = False,
            holiday_region: Optional[List[str]] = None,  # Array STRING
            clean_spikes_and_dips: bool = False,
            adjust_step_changes: bool = False,
            time_series_length_fraction: Optional[float] = None,
            min_time_series_length: Optional[int] = None,
            max_time_series_length: Optional[int] = None,
            trend_smoothing_window_size: Optional[int] = None,
            l2_reg: Optional[float] = None,
            kms_key_name: Optional[str] = None,

            # Optionale Spalten, die als Regression-Features verwendet werden sollen (XReg)
            feature_columns: Optional[List[str]] = None,
    ) -> bigquery.job.QueryJob:
        """
        Erstellt und trainiert ein BQML Zeitreihen-Modell (ARIMA_PLUS_XREG).
        """

        full_model_path = self.get_table_name(model_name)
        full_training_table_path = self.get_table_name(training_table)

        # 1. Sammle alle OPTIONS Parameter
        ml_options = {
            'model_type': model_type,
            'time_series_timestamp_col': time_series_timestamp_col,
            'time_series_data_col': time_series_data_col,
            'time_series_id_col': time_series_id_col,
        }

        # Füge optionale Parameter hinzu, falls sie gesetzt sind
        optional_params = {
            'horizon': horizon,
            'auto_arima': auto_arima,
            'auto_arima_max_order': auto_arima_max_order,
            'auto_arima_min_order': auto_arima_min_order,
            'non_seasonal_order': non_seasonal_order,
            'data_frequency': data_frequency,
            'include_drift': include_drift,
            'holiday_region': holiday_region,
            'clean_spikes_and_dips': clean_spikes_and_dips,
            'adjust_step_changes': adjust_step_changes,
            'time_series_length_fraction': time_series_length_fraction,
            'min_time_series_length': min_time_series_length,
            'max_time_series_length': max_time_series_length,
            'trend_smoothing_window_size': trend_smoothing_window_size,
            'l2_reg': l2_reg,
            'kms_key_name': kms_key_name,
        }

        for k, v in optional_params.items():
            if v is not None:
                ml_options[k] = v

        # 2. SELECT Spalten definieren
        select_cols = [time_series_timestamp_col, time_series_data_col, time_series_id_col]

        # Füge die XREG (Exogene Regressoren) Spalten hinzu
        if feature_columns:
            # Stelle sicher, dass keine doppelten Spalten vorhanden sind
            select_cols.extend([col for col in feature_columns if col not in select_cols])
        select_clause=""
        for item in select_cols:
            select_clause +=  f", {item}"

        # 3. SQL-Abfrage erstellen
        options_sql = self._format_bqml_options(ml_options)
        options_sql=options_sql.lstrip(",\n")

        sql_query = f"""
        CREATE OR REPLACE MODEL `{full_model_path}`
        OPTIONS(
            {options_sql}
        ) AS
        SELECT
            {select_clause}
        FROM
            `{full_training_table_path}`
        """

        print("create model...")
        return self.bqclient.query(sql_query)  # Verwende direkt bqclient.query, da run_query den Job ausführt und wartet.



if __name__ == "__main__":
    ml_master = BQMLMaster(
        dataset_id=ENV_ID
    )
    ml_master.train_time_series_model()