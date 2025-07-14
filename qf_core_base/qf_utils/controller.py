import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

from qf_core_base.qf_utils.pos import pos_list


# Die gegebene Liste von Positionen.
# Jede innere Liste repräsentiert einen "Node" oder eine "Position" mit 3 Koordinaten.
# Ein "Schritt" bedeutet das Wechseln zum nächsten Node in der Liste.













class Controller(AsyncWebsocketConsumer):
    """
    Live Agent that:
    Starts and controlls each sim -> etra layer between frontend and

    Controlled over command line temrinal: python3 -m websockets ws://localhost:8000/ws



    Receives:
    Prompt / commands of change
    Creates Cfg
    Redirects to "relay_station"
    Listen on FB rtdb for real time
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pattern = 'default'  # Standard-Pattern
        self.current_step_index = 0  # Aktueller Index in der pos_list
        self.pos_list = pos_list  # Die zu verwendende Positionsliste
        self.is_streaming = False  # Flag für aktives Streaming
        self.streaming_task = None  # Hält die asyncio-Task für das Streaming


    async def send_json_to_client(self, data_dict):
        """Hilfsmethode zum Senden von JSON-Daten an den Client."""
        await self.send(text_data=json.dumps(data_dict))

    async def receive(self, text_data):
        """
        Empfängt Nachrichten vom WebSocket-Client.
        Basierend auf dem ausgewählten Pattern werden hier die Ergebnisse
        über jeden Schritt für jeden Node zurückgegeben und prozessiert.
        """
        try:
            data = json.loads(text_data)
            command = data.get('command')
            payload = data.get('payload', {})

           #print(f"Empfangener Befehl: {command} mit Payload: {payload}")

            if command == 'set_pattern':
                new_pattern = payload.get('pattern_name')
                # Überprüfen, ob das Pattern valide ist
                if new_pattern in ["default", "add_10_to_coords", "only_even_steps", "reverse_coords"]:
                    self.current_pattern = new_pattern
                    # Optional: Schrittindex zurücksetzen, wenn das Pattern geändert wird
                    # self.current_step_index = 0
                    await self.send_json_to_client({
                        "type": "pattern_set",
                        "status": "success",
                        "current_pattern": self.current_pattern,
                        "message": f"Pattern auf '{self.current_pattern}' gesetzt."
                    })
                else:
                    await self.send_json_to_client({
                        "type": "error",
                        "message": f"Unbekanntes Pattern: {new_pattern}"
                    })

            elif command == 'start_streaming':
                if not self.is_streaming:
                    self.is_streaming = True
                    # Startet am aktuellen Index oder setzt zurück, falls gewünscht
                    # self.current_step_index = 0
                    delay = float(payload.get('delay', 1.0))  # Verzögerung in Sekunden
                    self.streaming_task = asyncio.create_task(self._stream_positions_loop(delay))
                    await self.send_json_to_client({
                        "type": "streaming_started",
                        "message": f"Streaming gestartet mit Pattern '{self.current_pattern}' und {delay}s Verzögerung."
                    })
                else:
                    await self.send_json_to_client({
                        "type": "info",
                        "message": "Streaming ist bereits aktiv."
                    })

            elif command == 'stop_streaming':
                if self.is_streaming:
                    self.is_streaming = False
                    if self.streaming_task:
                        self.streaming_task.cancel()
                        # Auf Beendigung warten ist nicht unbedingt nötig, da das Flag is_streaming die Schleife stoppt
                    await self.send_json_to_client({
                        "type": "streaming_stopped",
                        "message": "Streaming gestoppt."
                    })
                else:
                    await self.send_json_to_client({
                        "type": "info",
                        "message": "Kein aktives Streaming zum Stoppen."
                    })

            elif command == 'next_step':
                # Verarbeitet und sendet den nächsten Schritt manuell
                if not self.is_streaming:  # Nur wenn nicht bereits gestreamt wird
                    await self._process_and_send_current_step()
                else:
                    await self.send_json_to_client({
                        "type": "info",
                        "message": "Streaming ist aktiv. 'next_step' ist während des Streamings deaktiviert."
                    })

            # Beispiel für client-seitige Daten, die prozessiert werden sollen
            elif command == 'process_client_data':
                client_node_data = payload.get('node_data')
                # Hier Logik einfügen, um client_node_data mit dem aktuellen
                # Node aus pos_list und dem aktuellen Pattern zu verarbeiten.
                # Dies ist ein Platzhalter und muss spezifisch implementiert werden.
                if self.current_step_index < len(self.pos_list):
                    current_server_node = self.pos_list[self.current_step_index]
                    processed_data = {
                        "client_input": client_node_data,
                        "server_node_at_step": current_server_node,
                        "applied_pattern": self.current_pattern,
                        "result": f"Daten '{client_node_data}' für Node {self.current_step_index} (Pattern: {self.current_pattern}) verarbeitet."
                    }
                    await self.send_json_to_client({
                        "type": "client_data_processed",
                        "step": self.current_step_index,
                        "data": processed_data
                    })
                else:
                    await self.send_json_to_client({
                        "type": "error",
                        "message": "Ende der Positionsliste erreicht oder Index ungültig."
                    })


            else:
                await self.send_json_to_client({
                    "type": "error",
                    "message": f"Unbekannter Befehl: {command}"
                })

        except json.JSONDecodeError:
            await self.send_json_to_client({"type": "error", "message": "Ungültiges JSON empfangen."})
        except Exception as e:
            await self.send_json_to_client({"type": "error", "message": f"Ein Fehler ist aufgetreten: {str(e)}"})
           #print(f"Fehler in receive: {e}")

    async def _apply_pattern_to_node(self, node_data, step_idx):
        """
        Wendet das aktuelle Pattern auf einen einzelnen Node an.
        Gibt das prozessierte Node-Datum oder None zurück, falls es nicht gesendet werden soll.
        """
        if self.current_pattern == 'default':
            return node_data
        elif self.current_pattern == 'add_10_to_coords':
            return [coord + 10 for coord in node_data]
        elif self.current_pattern == 'only_even_steps':
            return node_data if step_idx % 2 == 0 else None
        elif self.current_pattern == 'reverse_coords':
            return node_data[::-1]  # Kehrt die Reihenfolge der Koordinaten um
        # Fügen Sie hier weitere Patterns hinzu
        # elif self.current_pattern == 'mein_neues_pattern':
        #     # Ihre Logik hier
        #     return ...
        else:
            # Fallback für ein unbekanntes (aber valide gesetztes) Pattern
           #print(f"Warnung: Pattern '{self.current_pattern}' hat keine Implementierung in _apply_pattern_to_node.")
            return node_data  # Oder None, oder Fehlerobjekt

    async def _process_and_send_current_step(self):
        """
        Verarbeitet den aktuellen Schritt basierend auf dem Pattern und sendet die Daten.
        Rückgabe: True, wenn ein Schritt gesendet wurde, False wenn das Ende erreicht ist oder nichts gesendet wurde.
        """
        if self.current_step_index >= len(self.pos_list):
            await self.send_json_to_client({
                "type": "info",
                "step": self.current_step_index,
                "message": "Ende der Positionsliste erreicht."
            })
            # Optional: Streaming stoppen, wenn es aktiv war und das Ende erreicht ist
            if self.is_streaming:
                self.is_streaming = False  # Stoppt die _stream_positions_loop
            return False

        node_data_original = self.pos_list[self.current_step_index]
        processed_node = await self._apply_pattern_to_node(node_data_original, self.current_step_index)

        data_sent_this_step = False
        if processed_node is not None:
            await self.send_json_to_client({
                "type": "position_update",
                "pattern": self.current_pattern,
                "step": self.current_step_index,
                "node_id": self.current_step_index,  # Annahme: Node-ID ist der Index
                "data": processed_node,
            })
            data_sent_this_step = True

        self.current_step_index += 1  # Zum nächsten Schritt für den nächsten Aufruf wechseln
        return data_sent_this_step  # Oder True, wenn der Index erhöht wurde, auch wenn nichts gesendet wurde (Pattern-abhängig)

    async def _stream_positions_loop(self, delay: float):
        """Interne Schleife für das kontinuierliche Senden von Positionen."""
       #print(f"Streaming-Schleife gestartet mit Index {self.current_step_index}, Pattern '{self.current_pattern}'.")
        try:
            while self.is_streaming:
                if self.current_step_index >= len(self.pos_list):
                    await self.send_json_to_client({
                        "type": "streaming_ended",
                        "message": "Streaming beendet: Ende der Positionsliste erreicht."
                    })
                    self.current_step_index = 0  # Optional: Zurücksetzen für nächsten Stream-Start
                    break  # Schleife verlassen

                await self._process_and_send_current_step()
                # Kurze Pause, um den Event-Loop nicht zu blockieren und CPU-Last zu reduzieren
                # Auch wenn `_process_and_send_current_step` `await` enthält,
                # sorgt `asyncio.sleep(0)` dafür, dass andere Tasks laufen können.
                # Für eine tatsächliche Verzögerung den `delay`-Wert verwenden.
                if delay > 0:
                    await asyncio.sleep(delay)
                else:  # Mindestens ein Yield Point, um andere Tasks nicht zu blockieren
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            print("Cancelled operation")
            #print("Streaming-Schleife wurde abgebrochen.")
            # Senden einer Nachricht über den Abbruch ist optional, da disconnect() oder stop_streaming() dies auslösen
        except Exception as e:
           #print(f"Fehler in der Streaming-Schleife: {e}")
            await self.send_json_to_client({"type": "error", "message": f"Streaming-Fehler: {str(e)}"})
        finally:
            self.is_streaming = False  # Sicherstellen, dass das Flag zurückgesetzt wird
            self.streaming_task = None  # Task-Referenz entfernen
           #print("Streaming-Schleife beendet.")
