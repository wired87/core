from manim import *
import numpy as np

# --- 1. Annahme: Datenstruktur für Fermion-Knoten aus deinem Simulator
#    Ersetze dies mit dem tatsächlichen Übergabemechanismus deiner Sim.
#    Im echten Fall würden diese Daten aus deinem laufenden Simulationsobjekt kommen.
#    Hier simulieren wir sie für die Manim-Darstellung.
class SimulatedFermionNode:
    def __init__(self, id, pos, is_quark=False, mass=0.1):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.is_quark = is_quark
        self.m = mass # Example mass

        # Initialize psi: For simplicity, let's make it a random complex spinor/color-spinor
        if is_quark:
            # (3,4) for quarks (3 color components, 4 spinor components)
            self.psi = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
            # Normalize for visualization purposes, make magnitude between 0 and 1
            self.psi = self.psi / np.linalg.norm(self.psi) * 0.5
        else:
            # (4,1) for leptons (4 spinor components)
            self.psi = np.random.rand(4, 1) + 1j * np.random.rand(4, 1)
            # Normalize for visualization purposes
            self.psi = self.psi / np.linalg.norm(self.psi) * 0.5

        # Initialize dpsi (a list of 4 arrays: [d_t_psi, d_x_psi, d_y_psi, d_z_psi])
        # These would come from your _dpsi method
        self.dpsi = []
        for _ in range(4): # For t, x, y, z
            if is_quark:
                self.dpsi.append(np.random.rand(3, 4) + 1j * np.random.rand(3, 4))
            else:
                self.dpsi.append(np.random.rand(4, 1) + 1j * np.random.rand(4, 1))

    # Helper to get a scalar magnitude for visualization
    def get_psi_magnitude(self):
        return np.linalg.norm(self.psi) # L2 norm of the entire (color-)spinor

    # Helper to get a simplified 3D "gradient" for visualization
    # This is a HEAVY simplification for display purposes only.
    # We are taking the real part of the 0th component of the spatial derivatives.
    def get_simplified_spatial_derivative_vector(self):
        # dpsi[1] is dx_psi, dpsi[2] is dy_psi, dpsi[3] is dz_psi
        # We take the real part of the 0th spinor component (and 0th color component if quark)
        if self.is_quark:
            # For quarks, take the first color component's first spinor component's real part
            dx_comp = self.dpsi[1][0, 0].real
            dy_comp = self.dpsi[2][0, 0].real
            dz_comp = self.dpsi[3][0, 0].real
        else:
            # For leptons, take the first spinor component's real part
            dx_comp = self.dpsi[1][0, 0].real
            dy_comp = self.dpsi[2][0, 0].real
            dz_comp = self.dpsi[3][0, 0].real

        return np.array([dx_comp, dy_comp, dz_comp])

# --- 2. Die Manim Visualizer Klasse
class FermionFieldVisualizer(ThreeDScene):
    def construct(self):
        # Setup camera and axes
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-5, 5, 1],
            x_length=10, y_length=10, z_length=10
        )
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        self.add(axes)
        self.add_labels(axes) # Optional: Add axis labels

        # Create simulated fermion nodes (replace with actual admin_data from your simulator)
        # We will create a small grid of nodes
        fermion_nodes = []
        grid_step = 2.0
        grid_range = 2.0
        node_id_counter = 0
        for x in np.arange(-grid_range, grid_range + 0.1, grid_step):
            for y in np.arange(-grid_range, grid_range + 0.1, grid_step):
                for z in np.arange(-grid_range, grid_range + 0.1, grid_step):
                    pos = np.array([x, y, z])
                    # Alternate between lepton and quark for demonstration
                    is_quark = (node_id_counter % 2 == 0)
                    fermion_nodes.append(SimulatedFermionNode(f"node_{node_id_counter}", pos, is_quark))
                    node_id_counter += 1

        # Mobjects to store for animation
        nodes_mobjects = VGroup()
        arrows_mobjects = VGroup()
        labels_mobjects = VGroup()

        # Visualize each fermion node
        for node in fermion_nodes:
            # Node position
            pos = node.pos

            # --- Visualize psi (magnitude)
            psi_mag = node.get_psi_magnitude()
            # Map magnitude to radius and color (e.g., larger radius for stronger field)
            # Normalize magnitude to a reasonable range for radius and color
            max_mag = 0.5 # Based on how we normalized psi in SimulatedFermionNode
            radius = 0.1 + psi_mag / max_mag * 0.2 # Min radius 0.1, max 0.3
            color_intensity = np.clip(psi_mag / max_mag, 0, 1) # 0 to 1 for color mapping

            # Use different colors for leptons/quarks and intensity for magnitude
            if node.is_quark:
                # Quarks as purples, increasing intensity
                node_color = ManimColor.interpolate(PURPLE_A, PURPLE_D, color_intensity)
            else:
                # Leptons as greens, increasing intensity
                node_color = ManimColor.interpolate(GREEN_A, GREEN_D, color_intensity)

            sphere = Sphere(center=pos, radius=radius, color=node_color, resolution=(20, 20))
            nodes_mobjects.add(sphere)

            # --- Visualize dmu_psi (simplified 3D vector representing spatial derivative)
            simplified_grad_vector = node.get_simplified_spatial_derivative_vector()
            # Scale the arrow length for better visibility
            arrow_scale = 1.0 # Adjust as needed
            arrow_start = pos
            arrow_end = pos + arrow_scale * simplified_grad_vector

            # Only draw arrow if gradient is significant
            if np.linalg.norm(simplified_grad_vector) > 1e-6:
                arrow = Arrow3D(arrow_start, arrow_end, color=BLUE, thickness=0.01) # Small thickness
                arrows_mobjects.add(arrow)

            # --- Optional: Add labels
            label_text = ""
            if node.is_quark:
                label_text += "Q "
            else:
                label_text += "L "
            label_text += f"id:{node.id}"
            label = Text(label_text, font_size=16).next_to(sphere, UP * 0.5 + RIGHT * 0.5)
            # labels_mobjects.add(label) # Uncomment to add labels, can clutter small grids

        self.play(Create(nodes_mobjects), Create(arrows_mobjects)) # Labels can be added here too

        # Hold the scene for a bit
        self.wait(3)

        # Optional: Animate a rotation to see from different angles
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()


    def add_labels(self, axes):
        # Add labels for axes
        labels = VGroup(
            Text("X").next_to(axes.get_x_axis().get_end(), RIGHT * 0.5),
            Text("Y").next_to(axes.get_y_axis().get_end(), UP * 0.5),
            Text("Z").next_to(axes.get_z_axis().get_end(), OUT * 0.5)
        )
        self.add(labels)


if __name__ == "__main__":
    fv=FermionFieldVisualizer()
    fv.construct()