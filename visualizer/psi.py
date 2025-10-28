from typing import Any

import numpy as np
import matplotlib

from qf_sim._qutip.visualizer import QuTiPRenderer

matplotlib.use("Agg")
qutip_renderer = QuTiPRenderer()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_plot_world(scale, grid_size):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Erstelle ein 3D-Gitter für die Visualisierung
    x = np.linspace(-scale, scale, grid_size)
    y = np.linspace(-scale, scale, grid_size)
    z = np.linspace(-scale, scale, grid_size)
    return x,y,z,ax,fig



def visualize_field_evolution_3d(field_data: list[Any], field_name, node_id, save_path, save, buffer, grid_size=30, scale=1.0):
    """
    Visualisiert die zeitliche Entwicklung von Feldwerten als 3D-Animation.
    """

    x = np.linspace(-scale, scale, grid_size)
    y = np.linspace(-scale, scale, grid_size)
    z = np.linspace(-scale, scale, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter([], [], [], c=[], cmap='viridis', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D-Feldentwicklung für {node_id}')

    qutip_renderer = QuTiPRenderer()  # Initialisiere Renderer

    def update(frame):
       #print(f"Run frame {frame} for id {node_id}")
        field_value = field_data[frame]

        # Render über QuTiP
        qutip_renderer.render(
            node_id,
            field_name,
            field_value,
        )

        # Dichte berechnen (wenn komplexer Tensor)
        if isinstance(field_value, np.ndarray):
            probability_density = np.abs(field_value)**2
            total_amplitude_sq = np.sum(probability_density)

            threshold = 0.005 * total_amplitude_sq
            mask = probability_density > threshold

            try:
                scat._offsets3d = (X[mask], Y[mask], Z[mask])
                scat.set_array(probability_density[mask].flatten())
                scat.set_clim(vmin=0, vmax=np.max(probability_density))  # Farbskala
            except:
                pass  # Falls shape nicht matcht

        ax.set_title(f'{node_id} – Field {field_name} - Frame: {frame}')
        return scat,

    ani = FuncAnimation(fig, update, frames=len(field_data), blit=False, interval=200)

    if save:
        ani.save(save_path, writer='ffmpeg', dpi=150)
    else:
        plt.show()

