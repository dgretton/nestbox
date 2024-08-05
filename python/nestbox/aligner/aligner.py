import numpy as np
from ..numutil import coerce_numpy, coerce_quaternion
from ..coordsystem import CoordinateSystem
from ..interfaces import AlignmentResultInterface
from typing import List, Union
import time
from threading import RLock

class Aligner:
    def __init__(self):
        self.coordinate_systems = []
        self.name_map = {}
        # name everything so it can show up in the computation graph
        self.current_origins = []
        self.current_orientations = []
        self.loss = None # negative log likelihood output of the model with respect to which we will compute gradients, created in build_model
        self.learning_rate_factor = 1.0
        self.losses = []
        self.pinned_cs_idx = None
        self.lock = RLock()

    def add_coordinate_system(self, coord_sys: CoordinateSystem, initial_origin=None, initial_orientation=None):
        with self.lock:
            if initial_origin is None:
                initial_origin = np.zeros(3)
            if initial_orientation is None:
                initial_orientation = np.array([1, 0, 0, 0])
            self.current_origins.append(coerce_numpy(initial_origin))
            self.current_orientations.append(coerce_numpy(initial_orientation))
            self.coordinate_systems.append(coord_sys)
            self.name_map[coord_sys.name] = coord_sys

    def get_coordinate_system(self, name):
        with self.lock:
            return self.name_map[name]

    def delete_coordinate_system(self, name):
        with self.lock:
            cs = self.get_coordinate_system(name)
            idx = self.coordinate_systems.index(cs)
            self.coordinate_systems.pop(idx)
            self.name_map.pop(name)
            self.current_origins.pop(idx)
            self.current_orientations.pop(idx)

    def clear_empty_coordinate_systems(self):
        with self.lock:
            print("Clearing any empty coordinate systems")
            for cs in self.coordinate_systems:
                print(f"Coordinate system {cs.name} has {len(cs.measurements)} measurements")
                if len(cs.measurements) == 0:
                    print(f"Deleting empty coordinate system {cs.name}")
                    self.delete_coordinate_system(cs.name)

    def reset_coordinate_system(self, coord_sys, set_origin, set_orientation):
        with self.lock:
            for i, cs in enumerate(self.coordinate_systems):
                if cs is coord_sys:
                    self.current_origins[i] = coerce_numpy(set_origin)
                    self.current_orientations[i] = coerce_numpy(set_orientation)
                    coord_sys.set_stale(True)
                    return
            raise ValueError("Coordinate system not found in aligner")

    def pin(self, coord_sys):
        with self.lock:
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
            raise ValueError(f"Argument {coord_sys} of invalid type for pinning ({type(coord_sys)}). Must be a string, {CoordinateSystem}, or int.")

    def reset_pin(self):
        self.pinned_cs_idx = None

    def get_transform(self, source_cs, target_cs):
        if not isinstance(source_cs, CoordinateSystem):
            source_cs = self.get_coordinate_system(source_cs)
        if not isinstance(target_cs, CoordinateSystem):
            target_cs = self.get_coordinate_system(target_cs)
        source_idx = self.coordinate_systems.index(source_cs)
        target_idx = self.coordinate_systems.index(target_cs)
        with self.lock:
            source_o = coerce_numpy(self.current_origins[source_idx])
            target_o = coerce_numpy(self.current_origins[target_idx])
            source_q = coerce_quaternion(self.current_orientations[source_idx])
            target_q = coerce_quaternion(self.current_orientations[target_idx])
        displacement = source_q.inverse.rotate(target_o - source_o)
        rotation = source_q.inverse * target_q
        timestamp = str(time.time())
        return AlignmentResult(timestamp, status=0, origin=displacement, quaternion=rotation, delta_velocity=[0, 0, 0])

    def get_basis_change_transform(self, source_cs, target_cs):
        return self.get_transform(source_cs=target_cs, target_cs=source_cs)

    def iterate_coordinate_systems(self):
        for i, coord_sys in enumerate(self.coordinate_systems):
            yield coord_sys, self.current_origins[i], self.current_orientations[i]

    def stale(self):
        for coord_sys in self.coordinate_systems:
            if coord_sys.stale:
                return True
        return False

    def build_model(self):
        with self.lock:
            self._build_model()

    def gradient_descent_step(self):
        with self.lock:
            self._gradient_descent_step()

    def _build_model(self):
        pass

    def _gradient_descent_step(self):
        pass


class AlignmentResult(AlignmentResultInterface):
    def __init__(self, timestamp: str, status: int, origin: List[float],
                 quaternion: List[float], delta_velocity: Union[List[float], None] = None):
        self._timestamp = timestamp
        self._status = status
        self._origin = [float(x) for x in origin]
        self._quaternion = [float(q) for q in quaternion]
        if delta_velocity is None:
            self._delta_velocity = [0.0, 0.0, 0.0]
        else:
            self._delta_velocity = [float(v) for v in delta_velocity]

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def status(self):
        return self._status

    @property
    def matrix_transform(self):
        x, y, z, w = self.quaternion
        return [
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ]

    @property
    def origin(self):
        return self._origin

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def delta_velocity(self):
        return self._delta_velocity

    def to_json(self):
        return {
            'timestamp': self.timestamp,
            'status': self.status,
            'origin': self.origin,
            'quaternion': self.quaternion,
            'delta_velocity': self.delta_velocity
        }

    @classmethod
    def from_json(cls, data):
        return cls(data['timestamp'], data['status'], data['origin'], data['quaternion'], data['delta_velocity'])
