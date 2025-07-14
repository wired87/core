import numpy as np

from qf_sim.physics.quantum_fields.nodes.g.GaugeBase import GaugeBase


class Z_Boson(GaugeBase):

    """
    Nur der o operator ändert sich
    """
    def __init__(self, attrs):
        super().__init__(**attrs)


