import random

import ray

class RaySetup:
    """
    Kapselt die Initialisierung von Ray, die Graphenerstellung,
    die Actor-Verwaltung und den Simulationsablauf.
    """
    def __init__(self, g, num_cpus: int = None):
        self.g = g
        self.num_cpus = num_cpus
        self.node_actors = {}  # {node_id: actor_ref}
        self.sensor_nodes = []
        self.other_nodes = []

    def initialize_ray(self):
        """Initialisiert das Ray-Runtime-System."""
        if not ray.is_initialized():
            if self.num_cpus:
                ray.init(num_cpus=self.num_cpus)
            else:
                #_ray.init()
            print(f"Ray Dashboard: {ray.get_dashboard_url()}")
            print("Ray initialized.")
        else:
            print("Ray is already initialized.")

    def _filter_graph(self):
        # specifies Nodes for the _ray cluster
        self.sensor_nodes = [n for n, data in self.g.G.nodes(data=True) if data.get('type') == 'sensor']
        self.other_nodes = [n for n, data in self.g.G.nodes(data=True) if data.get('type') != 'sensor']
        print("Sample graph created.")

    def set_actor_neighbors(self):
        """Setzt die Nachbar-Referenzen für jeden Ray Actor."""
        if not self.node_actors:
            raise ValueError("Actors must be instantiated before setting neighbors.")

        print("Setting up actor neighbors...")
        for node_id, actor_ref in self.node_actors.items():
            neighbors_in_graph = list(self.graph.neighbors(node_id))
            neighbor_refs_for_actor = {
                n_id: self.node_actors[n_id] for n_id in neighbors_in_graph
            }
            ray.get(actor_ref.set_neighbors.remote(neighbor_refs_for_actor))
        print("Actor neighbors set.")

    def run_simulation_steps(self, simulation_steps: int):
        """Führt die Simulationsschritte aus."""
        if not self.node_actors:
            raise ValueError("Actors must be instantiated and neighbors set before running simulation.")

        print("\n--- Simulation Start ---")
        for step in range(simulation_steps):
            print(f"\n--- Simulation Step {step + 1} ---")

            # Kommunikation: Sensoren senden ihre Werte an Nachbarn
            send_futures = []
            for node_id, actor_ref in self.node_actors.items():
                futures_from_send = actor_ref.send_value_to_neighbors.remote()
                if futures_from_send:
                    send_futures.extend(futures_from_send)

            if send_futures:
                ray.get(send_futures)
                print(f"All {len(send_futures)} messages sent for step {step + 1}.")
            else:
                print("No sensor nodes sent messages in this step.")

            # Nachrichtenverarbeitung: Alle Actors verarbeiten ihre empfangenen Nachrichten
            process_futures = []
            for node_id, actor_ref in self.node_actors.items():
                process_futures.append(actor_ref.process_messages.remote())

            ray.get(process_futures)
            print(f"All nodes processed messages for step {step + 1}.")

            # Optional: Zufällige Wertänderung bei einigen Sensoren, um Dynamik zu simulieren
            if step < simulation_steps -1: # Nicht im letzten Schritt ändern
                for sensor_id in self.sensor_nodes:
                    if random.random() < 0.5: # 50% Chance auf Wertänderung
                        new_val = round(random.uniform(5.0, 20.0), 2)
                        ray.get(self.node_actors[sensor_id].update_value.remote(new_val))
                        print(f"NodeActor {sensor_id}: Randomly changed value to {new_val}")

            # Aktuelle Zustände aller Knoten abrufen
            current_states_futures = [actor_ref.get_info.remote() for actor_ref in self.node_actors.values()]
            current_states = ray.get(current_states_futures)

            print("\n--- Current Node States ---")
            for state in current_states:
                print(f"  {state['node_id']} ({state['node_type']}): Value={state['value']}, Msgs Received={state['received_messages_count']}")

        print("\n--- Simulation Complete ---")

    def shutdown_ray(self):
        """Fährt das Ray-Runtime-System herunter."""
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shut down.")
        else:
            print("Ray is not initialized.")