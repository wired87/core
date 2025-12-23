import tempfile

import matplotlib.pyplot as plt
import numpy as np
import ray
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import base64
from io import BytesIO

from qf_core_base import USER_ID
from core.sm import FermUtils
from core.sm.gauge.gauge_utils import GaugeUtils
from qf_utils.all_subs import ALL_SUBS, FERMIONS, G_FIELDS
from qf_utils.field_utils import FieldUtils

"""

self.valid_data_keys = [
            "d_phi", "h", "phi", "vev", "h_prev", "energy", "potential_energy_H",
            "total_energy_H", "mass", "lambda_h", "laplacian_h", "mu_squared_H", "nphi",
            "dmu_phi_neighbors", "psi", "psi_bar", "velocity",
            "laplacian", "psi_prev", "isospin", "charge",
            "j_nu", "F_mu_nu", "gauge_group", "spin", "A",
            "dmu_A", "Wp", "dmu_Wp", "Wm", "dmu_Wm", "Z", "dmu_Z", "G",
            "dmu_G"
        ]

"""


class VisualizerProcessor:
    def __init__(self):
        self.valid_data_keys = [
            "phi",
            "psi",
            "G",
            "Wp",
            "Wm",
            "Z",
            "A",
        ]



    def main(self):
        return

    def _extract_field_values(self, nodes, nid) :
        """
        Receive timedata of specific nodes
        Loop through each timestep attrs and
        extract valid keys ->
        append to field_time_data
        returns field_time_data
        """
        field_time_data = {}
        for attrs in nodes:
            for key, value in attrs.items():
                if key in self.valid_data_keys:
                    if key not in field_time_data:
                        field_time_data[key] = []
                    field_time_data[key].append(
                        value
                    )
       #print(f"All Fields for {nid} extracted")
        return field_time_data




class Visualizer(
    FermUtils,
    GaugeUtils,
    VisualizerProcessor,
):
    """
    For live and nromal visualisations and


    Data get written to tmp dir (save_dir)
    """

    def __init__(self, save_dir, host=None, qf_utils=None):
        super().__init__()
        self.frames = []  # Store field arrays
        #self.frontend=frontend
        self.user_id = USER_ID
        self.qf_utils = qf_utils
        self.anim_buffer = BytesIO()
        self.save_dir = save_dir
        self.host=host
        self.animation_folder_name="animations"
        self.plt_folder_name="plots"

        self.field_utils = FieldUtils()

        self.plt_dest = os.path.join(self.save_dir, self.plt_folder_name)

        os.makedirs(self.plt_dest, exist_ok=True)

       #print("FieldVisualizer initialized")

        self.content = {}

       #print("Visualizer initialized")



    def get_all_subs(self):
        if self.host is None:
            all_subs: dict = self.qf_utils.get_all_subs_list(
                "base_type",
                datastore=True,
                just_attrs=False,
                sort_for_types=True
            )
        else:
            all_subs = ray.get(GLOBAC_STORE["UTILS_WORKER"].get_all_subs_list.remote(

            ))

    def main(self, all_subs):
       #print("create field visuals form sorted_node_types:")

        for i, (ntype, nodes) in enumerate(all_subs.items()):
            save_path = os.path.join(
                self.save_dir,
                f"pixel{ntype.split('pixel')[-1]}",

            )

            os.makedirs(save_path, exist_ok=True)

            save_path = os.path.join(
                save_path,
                f"{ntype}.mp4"

            )















    def preprare_G(self, G):
        all_nodes = [(nid, attrs) for nid, attrs in G.nodes(data=True) if attrs.get("base_type") in [*ALL_SUBS, "PIXEL"]]
        node_type_classified = {}


        for nid, attrs in all_nodes:
            ntype = attrs.get("type")
            if ntype not in node_type_classified:
                node_type_classified[ntype] = []
            node_type_classified[ntype].append(attrs)

        #




    def create_animation(self, key, field_data, fps=10, edges=list[tuple]):
        """
        Create an animation from NetworkX graphs.
        Each node: a point in space.
        Each edge: a line showing the coupling_term.
        """
        save_dest = os.path.join(self.animation_dest, f"{key}.mp4")

        # Setup Figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Prepare empty artists
        scat = ax.scatter([], [], s=50, c=[])
        lines = []

        def update(frame_idx):
            ax.clear()
            # Get all nodes for current timestep
            current_step_data = []
            for k, v in field_data.items():
                current_step_data.append(v["admin_data"][frame_idx])

            # Node positions (you must have "pos" attribute)
            positions = [item["pos"] for item in current_step_data]
            xs, ys, zs = zip(*positions)

            # Update scatter
            scat = ax.scatter(xs, ys, zs, c="white", s=60)

            # Draw edges with coupling_term labels
            for src, trgt, attrs in edges:
                # Get nodes from edges
                src_attrs = field_data[src][frame_idx]
                trgt_attrs = field_data[trgt][frame_idx]

                coupling_term = attrs.get("coupling_term")

                #  get pos
                p1 = src_attrs["pos"]
                p2 = trgt_attrs["pos"]

                opacity = self.coupling_to_alpha(coupling_term)

                # Plot 3d edge
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color="blue",
                    alpha=opacity
                )

                # Mittelpunkt in 3D
                mid = (
                    (p1[0] + p2[0]) / 2,
                    (p1[1] + p2[1]) / 2,
                    (p1[2] + p2[2]) / 2
                )

                # Label in 3D
                ax.text(
                    mid[0],
                    mid[1],
                    mid[2],
                    f"{coupling_term:.2f}",
                    fontsize=6,
                    color="white",
                    ha="center"
                )

            ax.set_title(f"Frame {frame_idx}")
            ax.set_xlim(-350, 350)
            ax.set_ylim(-350, 350)
            ax.set_zlim(-350, 350)

            return [scat]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(field_data),
            interval=1000 // fps,
            blit=False
        )

       #print("Animation fertig.")
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
            # 2. Animation speichern
            ani.save(tmpfile.name, writer=FFMpegWriter(fps=10))

            # 3. Datei lesen
            tmpfile.seek(0)
            return tmpfile.read()

            # 4. Base64 kodieren
            #            video_base64 = base64.b64encode(video_bytes).decode("utf-8")



    def coupling_to_alpha(self, coupling):
        """
        Konvertiert beliebige Kopplung (float, komplex, Matrix) in einen Float
        und leitet daraus die Alpha-Deckkraft (0..1) ab.
        """
        # 1. Falls None
        if coupling is None:
            return 0.0

        # 2. Falls Skalar (float oder komplex)
        if np.isscalar(coupling):
            value = abs(coupling)

        # 3. Falls Array oder Matrix
        else:
            arr = np.array(coupling)
            # Gesamtnorm
            value = np.linalg.norm(arr)

        # 4. Normalisieren auf Bereich 0..1
        # (z.B. bei Maxwert 10, passt du den Teiler an)
        norm_value = min(value / 10.0, 1.0)

        return norm_value

    def _process_field_array(self, field_entry):
        """Convert complex tuples to |ψ| arrays."""
        arr = np.array(field_entry)
        abs_val = np.sqrt(np.square(arr[..., 0]) + np.square(arr[..., 1]))
        return abs_val

    def _save(self, field_data):
        if isinstance(field_data, list):
            self.frames.extend(field_data)
        else:
            self.frames.append(field_data)

    def _create_frame_bytes(self, field_data, ntype):
        """Create a static image from field admin_data and return it as bytes b64"""

       #print("_create_frame_bytes", field_data)
        ax = plt.axes(projection="3d")
        fig = plt
        # === Fall 1: Skalarfeld (phi) ===
        if ntype == "phi":
            img = np.full((1, 1), field_data, dtype=float)
           #print("phi img", img)

        # === Fall 2: Fermionen (psi) ===
        elif ntype in FERMIONS:
            img = self._convert_to_complex(com=field_data)
           #print("ferm img", img)

        # === Fall 3: Gluonenfelder (A^a_μ) ===
        elif ntype in G_FIELDS:
            img = self._convert_to_complex(com=field_data)
           #print("gfield img", img)
            img = img.reshape(1, -1)
           #print("gfield img reshape", img)

        else:
            raise ValueError(f"Unsupported field shape: {np.shape(field_data)}")
       #print("field_data, converted", img)
        # === Visualisierung ===
        #cax = ax.imshow(img, cmap='viridis')
        cax = ax.imshow(np.abs(img), cmap='viridis')

        fig.colorbar(cax)
        ax.set_title("Field Snapshot")
       #print("Axes set")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
       #print("Bytes created")
        return base64.b64encode(buf.read()).decode('utf-8')


    def extract_field_values(self, sorted_node_types):
        ##pprint.pp(sorted_node_types)
        field_values: dict[list] = {}

        # Create field plots and animations for ALL
        for k, v in sorted_node_types.items():
            first_list_item = v[0]

            # Extract key of field value
            base_type = first_list_item["base_type"]
            #print("base_type", base_type)

            # Check for gauge and extract its specific field value key
            parent = first_list_item["parent"][0].lower()
            #print("vmain parent", parent)

            if parent == "gauge":
                # get field value name of g sub-field
                field_key = self._field_value(base_type.lower())

            elif parent == "phi":
                field_key = "h"
            else:
                # jsut lower @ psi -> gauge different
                field_key = parent.lower()
            #print("field_key", field_key)

            # Extrat just field values from nodes
            for item in v:
                if k not in field_values:
                    field_values[k] = []
                field_values[k].append(item[field_key])
        return field_values