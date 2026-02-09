import json

from gem_core.gem import Gem
from utils.logger import LOGGER

class AIChatClassifier:
    """
    Initialisiert den Chatbot mit API-Schlüssel und Konfigurationspfaden.
    Erstellt Konfigurationsdateien, falls nicht vorhanden.
    """
    def __init__(self, case_struct:list, cfg_creator=None):
        self.cfg_creator = cfg_creator
        self.case_struct = case_struct
        self.gem = Gem()
        self.user_history = {}
        self.goal: str or None = None # case of relay case struct


    def _extract_fields(self):
        prompt = f"""
        Given is a list of FIELDS.
        Your tasks is to align the given INPUT to all FIELDS to identify matches, 
        Return identified FIELD, separated by comma(,) - nothing else. 
        If you cant identify anything, return the word 'NA'.
        """
        fields =None# ask_gem(prompt)
        if 'NA' not in fields and ',' in fields:
            fields_list = fields.split(",")
            self.cfg_creator.add_fields(fields=fields_list)



    def _update_history(self, user_id, message):
        """Aktualisiert die Historie für einen Benutzer."""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append(message)

    def _classify_input(self, user_input, cases):
        """
        Klassifiziert die Benutzereingabe mithilfe der Gemini API für QFS-Befehle.
        Gibt den passenden Usecase-Schlüssel zurück oder 'general_qfs_query' als Standard.
        """
        print("_classify_input...")
        prompt = f"""Klassifiziere die folgende Benutzereingabe in einen der folgenden Usecases für die Steuerung einer Quantenfeld-Simulation: 
            {', '.join(cases)}. Gib nur den Usecase-Namen zurück. 
            Eingabe: '{user_input}'")
            """
        try:
            classification = self.gem.ask(prompt)
            if classification and isinstance(classification, str):
                classification = classification.strip().lower()
                if classification in cases:
                    print(f"Classification: {classification}")
                    return classification
        except Exception as e:
            print(f"Fehler bei der Klassifizierung: {e}")
        print(f"Classification not valid")
        return None

    def _give_info(self, user_input):
        """
        Klassifiziert die Benutzereingabe mithilfe der Gemini API für QFS-Befehle.
        Gibt den passenden Usecase-Schlüssel zurück oder 'general_qfs_query' als Standard.
        """
        prompt = (
            f"Beantworte die frage des Benutzers: "
            f"{', '.join(self.use_cases.keys())}. Wenn du die antwort i den folgenden docs nicht findest return einfach Einen netten satz dass dz die : "
            f"Eingabe: '{user_input}'")
        try:
            response = self.model.generate_content(prompt)
            friendly_response = response.text.strip().lower()
            if friendly_response:
                return friendly_response
        except Exception as e:
            print(f"Fehler bei der Klassifizierung: {e}")
        return "Hi, I have a lot to do currently. Please try again later"

    # --- QFS Usecase Handler Methoden ---
    def _handle_start_simulation(self, user_id, user_input):
        """Behandelt den Befehl zum Starten der Simulation."""
        if self.simulation_state["running"]:
            return "Die Simulation läuft bereits."
        else:
            self.simulation_state["running"] = True
            return f"Starte die Quantenfeld-Simulation mit aktuellen Parametern: {self.simulation_state['parameters']}. Aktueller Schritt: {self.simulation_state['current_step']}."

    def _handle_set_parameters(self, user_id, user_input):
        """Behandelt das Setzen von Simulationsparametern."""
        # Beispiel: Versuch, Parameter aus dem Input zu extrahieren
        # Eine reale Implementierung würde hier komplexere Parsing-Logik erfordern
        extracted_params = {}
        if "time_step" in user_input:
            try:
                time_step = float(user_input.split("time_step=")[1].split(" ")[0])
                extracted_params["time_step"] = time_step
            except (ValueError, IndexError):
                pass
        if "lattice_size" in user_input:
            try:
                lattice_size = int(user_input.split("lattice_size=")[1].split(" ")[0])
                extracted_params["lattice_size"] = lattice_size
            except (ValueError, IndexError):
                pass

        if extracted_params:
            self.simulation_state["parameters"].update(extracted_params)
            return f"Parameter aktualisiert: {extracted_params}. Aktuelle Parameter: {self.simulation_state['parameters']}"
        return "Bitte gib an, welche Parameter du setzen möchtest (z.B. 'set time_step=0.01')."

    def _handle_check_status(self, user_id, user_input):
        """Überprüft den aktuellen Status der Simulation."""
        status = "läuft" if self.simulation_state["running"] else "gestoppt"
        return f"Simulationsstatus: {status}. Aktueller Schritt: {self.simulation_state['current_step']}. Parameter: {self.simulation_state['parameters']}"

    def _handle_save_state(self, user_id, user_input):
        """Speichert den aktuellen Zustand der Simulation (simuliert)."""
        # Hier könnte die Logik zum Serialisieren und Speichern des Zustands stehen
        filename_match = [part for part in user_input.split() if ".json" in part]
        filename = filename_match[0] if filename_match else "simulation_state.json"

        try:
            with open(filename, 'w') as f:
                json.dump(self.simulation_state, f, indent=4)
            return f"Simulationszustand erfolgreich in '{filename}' gespeichert."
        except Exception as e:
            return f"Fehler beim Speichern des Zustands: {e}"

    def _handle_load_state(self, user_id, user_input):
        """Lädt einen Simulationszustand (simuliert)."""
        filename_match = [part for part in user_input.split() if ".json" in part]
        filename = filename_match[0] if filename_match else "simulation_state.json"

        try:
            with open(filename, 'r') as f:
                loaded_state = json.load(f)
                self.simulation_state.update(loaded_state)  # Aktualisiere den Zustand
            return f"Simulationszustand erfolgreich aus '{filename}' geladen. Der aktuelle Zustand ist: {self.simulation_state}"
        except FileNotFoundError:
            return f"Fehler: Datei '{filename}' nicht gefunden."
        except Exception as e:
            return f"Fehler beim Laden des Zustands: {e}"

    def _handle_plot_results(self, user_id, user_input):
        """Generiert Plots der Simulationsergebnisse (simuliert)."""
        # Eine reale Implementierung würde hier Plot-Befehle ausführen
        return "Generiere Visualisierungen der aktuellen Simulationsergebnisse. Welchen Typ Plot möchtest du sehen?"

    def _handle_stop_simulation(self, user_id, user_input):
        """Stoppt die Simulation."""
        if not self.simulation_state["running"]:
            return "Die Simulation ist bereits gestoppt."
        else:
            self.simulation_state["running"] = False
            return "Die Quantenfeld-Simulation wurde gestoppt."

    def _handle_general_qfs_query(self, user_id, user_input):
        """Behandelt allgemeine Fragen zur Quantenfeld-Simulation mithilfe von Gemini."""
        prompt = f"""Du bist ein asistent für eine quantensimulationssoftware unter folgendem url. Antworte auf die folgende allgemeine Frage zur Quantenfeld-Simulation: '{user_input}'."          
        Berücksichtige die Benutzerhistorie (falls vorhanden): {self.user_history.get(user_id, [])}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Fehler bei der Beantwortung der allgemeinen QFS-Anfrage: {e}")
            return self.config.get("default_qfs_response")

    def chat(self, user_id, user_input):
        """
        Hauptmethode zum Verarbeiten von Benutzereingaben.
        """
        cases = [c["case"] for c in self.case_struct]
        self._update_history(user_id, user_input)
        classification = self._classify_input(user_input, self.case_struct)
        handler_struct = self.case_struct[cases.index(classification)]

        if self.goal == None:
            self.goal = classification
            self.goal_data_struct = handler_struct["req_struct"]
        else:
            # question = follow up
            # -> collect more information for the struct
            prompt = f"""
            Extract as much information from the provided user query to fill the provided data structure.
             
             GOAL:
             {self.goal}
             
             DATA STRUCTURE:
             {self.goal_data_struct}
             
             USER QUERY:
             {user_input}
             
             MARK:
             return just the adapted DATA STRUCTURE in valid (stringified) python synthax.
             Add no etra chars
            
            AVOID THE FOLLOWING ERRORS:
            
            """
            response = self.gem.ask(prompt)
            # convert & save response
            try:
                updated_struct = json.loads(response)
                self.goal_data_struct.update(updated_struct)

                # make request
                for item in self.case_struct:
                    if item["case"] == self.goal:
                        # make request
                        response = item["func"](
                            paylaod=self.goal_data_struct
                        )

                        # if response cmpletes withoit err:reset goal and data struct
                        self.goal = None
                        self.goal_data_struct = {}

                        # return repsonse to user
                        return response
            except Exception as e:
                print("unable to save changes in response:", e)

        # set as gloabl goal
        handler = self.use_cases.get(classification, self._handle_general_qfs_query)
        response = handler(user_id, user_input)

        self._update_history(user_id, response)  # Antwort auch zur Historie hinzufügen
        return response


if __name__ == "__main__":
    chat_classifier = AIChatClassifier(case_struct=RELAY_CASES_CONFIG)
    response = chat_classifier.chat(
        self.user_id,
        user_input=payload["data"]["msg"],
    )