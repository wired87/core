from typing import Any
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from qutip import wigner, Qobj
matplotlib.use("TkAgg")



class QuTiPRenderer:
    def __init__(self):
        self.colorbar = None
        self.fig = plt.figure(figsize=(10, 8))

        # 2D Wigner
        self.ax2d = self.fig.add_subplot(
            1, # row
            2, # cols
            1,
        )

        # 3D Wigner
        self.ax3d = self.fig.add_subplot(1, 2, 2, projection="3d")

        self.scat = self.ax2d.scatter([], [], [], c=[], cmap='viridis', alpha=0.3)


    def render(self, node_id, field_name="psi", data=np.array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]), index=0): #basis(20, 0)
        print("node_id:", node_id)
        print("Field Name:", field_name)
        print("Field Data:", data)

        if field_name in ["psi"]:
            w_data = self._quark_data(
                node_id,
                data
            )
            print("Qobj Data:", w_data)

            self._wigner(
                w_data,
                node_id,
                index,
                is_quark="quark" in node_id.lower(),
            )

            """elif field_name in ["phi", "h"]:
                self.plot_tensor_field(np.abs(data), title=field_name)
    
            elif field_name in ["F_mu_nu", "d_phi", "dmu_phi_neighbors"]:
                self.plot_tensor_field(np.real(data), title=field_name)
            """
        else:
            print(f"⚠️ Kein Renderer für {field_name}")
            return None


    def _quark_data(
            self,
            node_id,
            data
    ):
        w_data = None
        is_quark = "quark" in node_id.lower()

        if isinstance(data, (np.ndarray, list)):
            if is_quark is True:
                w_data = []
                for i in range(3):
                    w_data.append(Qobj(data[i]))
            else:
                w_data = Qobj(data)
        return w_data





    def visualize(
            self,
            field_data: list[Any],
            field_name,
            node_id,
            save_path,
            save,
    ):
        """
        Visualisiert die zeitliche Entwicklung von Feldwerten als 3D-Animation.
        """
        def update(frame):
            print(f"Run frame {frame} for id {node_id}")
            field_value = field_data[frame]

            # Render über QuTiP
            self.render(
                node_id,
                field_name,
                field_value,
                index=frame
            )

            # Dichte berechnen (wenn komplexer Tensor)

        ani = FuncAnimation(self.fig, update, frames=len(field_data), blit=False, interval=600)

        if save:
            ani.save(save_path, writer='ffmpeg', dpi=150)
        else:
            plt.show()

    def _wigner(self, psi, node_id, index, is_quark):
        xvec = np.linspace(-6, 6, 200)
        yvec = np.linspace(-6, 6, 200)
        if is_quark is True:
            W = sum(
                wigner(psi[i], xvec, yvec)
                for i in range(3)
            )
        else:
            W = wigner(psi, xvec, yvec)

        print("W", W)

        X, Y = np.meshgrid(xvec, yvec)

        im = self.ax2d.contourf(X, Y, W, 100, cmap="RdBu")
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax2d)
        else:
            self.colorbar.on_mappable_changed(im)


        self.ax2d.set_title(f"{node_id} 2D Frame {index}")

        # 3D Wigner
        self.ax3d.plot_surface(X, Y, W, cmap="RdBu", edgecolor="none", alpha=0.9)
        self.ax3d.set_title(f"{node_id} 3D Frame {index}")

        return self._set_scatter(
            psi,
            X,
            Y,
            W
        )

    def _set_scatter(
            self,
            field_value,
            X,
            Y,
            W
    ):
        if isinstance(field_value, np.ndarray):
            probability_density = np.abs(field_value) ** 2
            total_amplitude_sq = np.sum(probability_density)

            threshold = 0.005 * total_amplitude_sq
            mask = probability_density > threshold
            try:
                self.scat._offsets3d = (X[mask], Y[mask], W[mask])
                self.scat.set_array(probability_density[mask].flatten())
                self.scat.set_clim(vmin=0, vmax=np.max(probability_density))  # Farbskala
            except:
                pass  # Falls shape nicht matcht
        return self.scat,





if __name__ == "__main__":
    renderer = QuTiPRenderer()
    renderer.render(node_id="hi")

    #plot_wigner_2d_3d(psi)
    plt.show()