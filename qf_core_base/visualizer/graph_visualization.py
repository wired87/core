import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
import io
import zipfile

# === KONFIGURATION ===
NODE_SIZE = 150
NODE_EDGE_COLOR = 'black'
NODE_EDGE_WIDTH = 0.5

BASE_EDGE_COLOR = (0.0, 0.5, 0.5)  # Türkis als Startfarbe für Edges

LINE_WIDTH_MIN = 0.5
LINE_WIDTH_MAX = 3.0
LINE_ALPHA_MIN = 0.05
LINE_ALPHA_MAX = 1.0  # Vollständig deckend bei voller Aktivität

FIGSIZE = (10, 8)
BACKGROUND_COLOR = "black"


# === HILFSFUNKTIONEN ===

def _compute_node_activity(graph):
    """Berechnet die Aktivität jedes Knotens als Summe seiner Kanten-Gewichte."""
    node_activity = np.zeros(len(graph.nodes()))
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 0)
        node_activity[u] += weight
        node_activity[v] += weight
    return node_activity


def _get_node_colors(node_activity, max_activity=1.0):
    """Berechnet Node-Farben von Weiß (0 Aktivität) nach Blau (max. Aktivität)."""
    if len(node_activity) == 0 or np.all(node_activity == 0):
        return [(1.0, 1.0, 1.0)] * len(node_activity)  # Alle weiß

    normalized_activity = node_activity / max_activity
    colors = []
    for n in normalized_activity:
        colors.append((1.0 - n, 1.0 - n, 1.0))  # Interpolation von Weiß zu Blau
    return colors


def _get_edge_style(weight, weight_min, weight_max):
    """Berechnet Transparenz (Alpha), Breite und Farbe der Kanten."""
    if weight_max == weight_min:
        norm = 0.0
    else:
        norm = (weight - weight_min) / (weight_max - weight_min)
    norm = np.clip(norm, 0, 1)

    alpha = LINE_ALPHA_MIN + (LINE_ALPHA_MAX - LINE_ALPHA_MIN) * norm

    # Interpolation von BASE_EDGE_COLOR zu Weiß
    r = BASE_EDGE_COLOR[0] * (1 - norm) + 1.0 * norm
    g = BASE_EDGE_COLOR[1] * (1 - norm) + 1.0 * norm
    b = BASE_EDGE_COLOR[2] * (1 - norm) + 1.0 * norm
    edge_color = (r, g, b)

    lw = LINE_WIDTH_MIN + (LINE_WIDTH_MAX - LINE_WIDTH_MIN) * norm

    return alpha, lw, edge_color


# === HAUPTMETHODE FÜR RENDERING ===

def render_field_graph_3d(G: nx.Graph,
                          positions: np.ndarray,
                          frames: int = 100,
                          interval: int = 100,
                          weight_change_range: float = 0.5,
                          save_path: str = None):
    """
    Rendert eine 3D-Animation eines Graphen mit dynamischen Node-Farben und Edge-Styles.
    """
    num_nodes = len(positions)
    if num_nodes == 0: return

    # Initiale Gewichts-Ranges für konsistente Skalierung
    initial_edge_weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    edge_weight_min = min(initial_edge_weights) if initial_edge_weights else 0.0
    edge_weight_max = max(initial_edge_weights) if initial_edge_weights else 1.0
    if edge_weight_max == edge_weight_min: edge_weight_max += 1e-6  # Vermeide Division durch Null

    initial_node_activity = _compute_node_activity(G)
    max_node_activity = np.max(initial_node_activity) if len(initial_node_activity) > 0 else 1.0
    if max_node_activity == 0: max_node_activity = 1.0

    # Figure und Achsen Setup
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_axis_off()

    ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
    ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
    ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())

    # Nodes initial rendern
    node_colors = _get_node_colors(initial_node_activity, max_node_activity)
    sc = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=NODE_SIZE, c=node_colors,
        edgecolors=NODE_EDGE_COLOR, linewidths=NODE_EDGE_WIDTH
    )

    # Edges initial rendern
    edge_refs = {}
    for i, j, data in G.edges(data=True):
        w = data.get('weight', 0)
        alpha, lw, color = _get_edge_style(w, edge_weight_min, edge_weight_max)
        line, = ax.plot([positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        [positions[i][2], positions[j][2]],
                        color=color, linewidth=lw, alpha=alpha)
        edge_refs[(i, j)] = line

    # Update-Funktion für die Animation
    def update(frame):
        # Gewichte der Edges aktualisieren (hier zufällig, ersetze mit deiner Logik)
        for i, j, data in G.edges(data=True):
            if 'weight' in data:
                delta = random.uniform(-weight_change_range, weight_change_range)
                data['weight'] = np.clip(data['weight'] + delta, edge_weight_min, edge_weight_max)
            else:
                data['weight'] = (edge_weight_min + edge_weight_max) / 2

        # Edges aktualisieren
        for (i, j), line in edge_refs.items():
            w = G[i][j].get('weight', 0)
            alpha, lw, color = _get_edge_style(w, edge_weight_min, edge_weight_max)
            line.set_alpha(alpha)
            line.set_linewidth(lw)
            line.set_color(color)

        # Nodes aktualisieren
        current_node_activity = _compute_node_activity(G)
        new_node_colors = _get_node_colors(current_node_activity, max_node_activity)
        sc.set_facecolor(new_node_colors)

        return sc, *list(edge_refs.values())

    # Animation starten oder speichern
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    if save_path:
       #print(f"Speichere Animation als {save_path}...")
        mp4_buffer = io.BytesIO()
        try:
            writer = animation.FFMpegWriter(fps=1000 / interval)
            ani.save(mp4_buffer, writer=writer, codec='h264')
            mp4_buffer.seek(0)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('animation.mp4', mp4_buffer.read())
            zip_buffer.seek(0)
            with open(save_path, "wb") as f:
                f.write(zip_buffer.getvalue())
           #print("Erfolgreich gespeichert.")
        except Exception as e:
           #print(f"Fehler beim Speichern: {e}. Ist ffmpeg installiert und im PATH?")
        finally:
            plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def create_subf_G(datastore):
    for time_step, nid in datastore.nodes(data=True):
