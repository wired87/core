import numpy as np
import jax.numpy as jnp


class Mover:
    def __init__(self, g):

        # Get realistic diffusion coefficient in µm²/s
        self.cell_index = 0
        self.target = None
        self.position = None
        self.g = g
        # Time step and movement step
        self.time_step = 0.001  # seconds

    def find_nearest_point_from_pos_list(self, start_pos, points):
        start = jnp.array(start_pos)
        points = jnp.array(points)

        distances = np.linalg.norm(points - start, axis=1)
        nearest_index = jnp.argmin(distances)
        return points[nearest_index], distances[nearest_index]

    def get_nearest_neighbors(self, start_pos, neighbors, amount_neighbors=8, pos_attr_key="pos"):
        """
        todo build in mechanism to treat corner
        and nodes @ the edge to vacuum
        """

        distances = []

        for neighbor in neighbors:
            if isinstance(neighbor, tuple):
                neighbor_id = neighbor[0]
                neighbor_attrs = neighbor[1]
            else:
                neighbor_id = neighbor
                neighbor_attrs = self.g.G.nodes[neighbor_id]

            pos = neighbor_attrs.get(pos_attr_key)
            if pos is None or neighbor_id == start_pos:
                continue

            distance = np.linalg.norm(np.array(pos) - jnp.array(start_pos))
            distances.append((distance, neighbor_id, neighbor_attrs))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Return top N
        result = [(nid, attrs) for _, nid, attrs in distances[:amount_neighbors]]

        print_result = []
        for r in result:
            print_result.append((r[0], r[1].get("pos")))

        #print("Neareest neighbors identified:", print_result)
        return result


    def move_src_to_trgt(self, pos1, pos2, step_size):
            direction = [p2 - p1 for p1, p2 in zip(pos1, pos2)]
            distance = sum(d ** 2 for d in direction) ** 0.5
            if distance == 0:
                return pos1  # already at the target
            unit_direction = [d / distance for d in direction]
            return [p1 + step_size * ud for p1, ud in zip(pos1, unit_direction)]



    def move(
            self,
            position_um,
            target_pos,
            source_radius,
            target_radius,
            diffusion,
    ):

        diffusion = diffusion

        step_size_um = np.sqrt(2 * diffusion * self.time_step)

        self.position = jnp.array(position_um, dtype=float)
        self.target = jnp.array(target_pos, dtype=float) if target_pos is not None else None

        if self.target is not None:
            direction = self.target - self.position
            dist = np.linalg.norm(direction)

            # Minimum allowed distance to prevent overlap
            min_dist = source_radius + target_radius

            # Already within safe distance — stop moving
            if dist <= min_dist:
                return

            direction = direction / dist
            move_vector = direction * step_size_um

            # Don't overshoot and pass the surface
            if dist - step_size_um < min_dist:
                move_vector = direction * (dist - min_dist)

        else:
            # Pure diffusion
            move_vector = np.random.normal(0, step_size_um, 3)

        self.position += move_vector

    def is_at_target(self, threshold=1.0):
        if self.target is None:
            return False
        return np.linalg.norm(self.target - self.position) < threshold


    def spread_objects(self, amount_items, dim, self_attrs):
        self.cell_index += 1

        # Use square cells for consistent spacing in x and y
        cols = int(amount_items ** 0.5)
        rows = (amount_items + cols - 1) // cols
        grid_size = min(dim / cols, dim / rows)  # enforce square spacing

        # 1-based to 0-based index
        index = self.cell_index - 1
        row = index // cols
        col = index % cols

        x = (col + 0.5) * grid_size
        y = (row + 0.5) * grid_size

        self_attrs["pos"] = [x, y, 0.0]


        #print(f"UPDATED POS {self.cell_index}:", self_attrs["pos"])
       #print(self_attrs["pos"])
        return self_attrs, grid_size

    def spread_objects_3d(
            self,
            amount_items,
            dim,
            self_attrs,
            spread_evenly: int or None = None
    ):
        #print(locals())
        try:
           #print("Spread items")
            self.cell_index += 1

            # Berechne Würfelgitter: gleich viele Zellen in x, y, z Richtung
            per_side = int(round(amount_items ** (1 / 3)))  # cube root
            if per_side ** 3 < amount_items:
                per_side += 1

            # Spread objects by a fixed number
            if spread_evenly is None:
                grid_size = dim / per_side
            else:
                grid_size = spread_evenly

            index = self.cell_index - 1
            x_idx = index % per_side
            y_idx = (index // per_side) % per_side
            z_idx = index // (per_side * per_side)

            x = x_idx * grid_size  # x_idx + .5
            y = y_idx * grid_size
            z = z_idx * grid_size

            self_attrs["pos"] = [x, y, z]

            #print(f"UPDATED POS {self.cell_index}:", self_attrs["pos"]
        except Exception as e:
            print(f"Error in spread_objects_3d: {e}")
        return self_attrs

    def distribute_subpoints_around_qfns(self, ds=20):
        print("set sub pos")
        for nid, attrs in self.g.G.nodes(data=True):
            if attrs.get("type").upper() == "PIXEL" and "pos" in attrs:
                base_pos = jnp.array(attrs["pos"])

                # Finde Subpunkte
                subs = self.g.get_neighbor_list(nid, trgt_rel="has_field")
                print(f"Subs for {nid}: {len(subs)}")
                if not subs:
                    continue

                # Gleichmäßig auf einer Kugel verteilen
                check_pos = []

                # Gleichmäßig auf Kugel verteilen (Fibonacci Sphere)
                count = len(subs)
                offset = 2.0 / count
                increment = np.pi * (3.0 - np.sqrt(5.0))  # Goldener Winkel

                for i, (sub_id, sub_attrs) in enumerate(subs):
                    y = ((i * offset) - 1) + (offset / 2)
                    r = np.sqrt(1 - y * y)
                    phi = i * increment

                    x = np.cos(phi) * r
                    z = np.sin(phi) * r

                    pos = base_pos + ds * jnp.array([x, y, z])
                    sub_attrs["pos"]=pos.tolist()
                    check_pos.append(
                        (sub_id,  sub_attrs["pos"])
                    )

                    self.g.update_node(sub_attrs)

                print(f"sub positions for {nid}:")
                for item in check_pos:
                    print(f"{item[0]}: {item[1]}")

        print("All sub pos set!")

