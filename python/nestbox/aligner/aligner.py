import numpy as np
from ..numutil import coerce_numpy

class Aligner:
    def __init__(self):
        self.coordinate_systems = []
        # name everything so it can show up in the computation graph
        self.current_origins = []
        self.current_orientations = []
        self.loss = None # negative log likelihood output of the model with respect to which we will compute gradients, created in build_model
        self.learning_rate_factor = 1.0
        self.losses = []
        self.pinned_cs_idx = None

    def add_coordinate_system(self, coord_sys, initial_origin=None, initial_orientation=None):
        if initial_origin is None:
            initial_origin = np.zeros(3)
        if initial_orientation is None:
            initial_orientation = np.array([1, 0, 0, 0])
        self.current_origins.append(coerce_numpy(initial_origin))
        self.current_orientations.append(coerce_numpy(initial_orientation))
        self.coordinate_systems.append(coord_sys)

    def reset_coordinate_system(self, coord_sys, set_origin, set_orientation):
        for i, cs in enumerate(self.coordinate_systems):
            if cs is coord_sys:
                self.current_origins[i] = coerce_numpy(set_origin)
                self.current_orientations[i] = coerce_numpy(set_orientation)
                coord_sys.set_stale(True)
                return
        raise ValueError("Coordinate system not found in aligner")

    def pin(self, coord_sys):
        if coord_sys is None or coord_sys == 'none':
            self.reset_pin()
            return
        if isinstance(coord_sys, str):
            for i, cs in enumerate(self.coordinate_systems):
                if cs.name == coord_sys:
                    self.pinned_cs_idx = i
                    return
        if isinstance(coord_sys, CoordinateSystem):
            coord_sys = self.coordinate_systems.index(coord_sys)
            return
        if isinstance(coord_sys, int):
            self.pinned_cs_idx = coord_sys
            return
        raise ValueError(f"Argument {coord_sys} of invalid type for pinning ({type(coord_sys)})")

    def reset_pin(self):
        self.pinned_cs_idx = None

    def iterate_coordinate_systems(self):
        for i, coord_sys in enumerate(self.coordinate_systems):
            yield coord_sys, self.current_origins[i], self.current_orientations[i]

    def stale(self):
        for coord_sys in self.coordinate_systems:
            if coord_sys.stale:
                return True
        return False

    def build_model(self):
        pass

    def gradient_descent_step(self, learning_rate=0.0001):
        pass
