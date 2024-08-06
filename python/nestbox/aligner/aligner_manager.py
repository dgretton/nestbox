from .aligner_factory import AlignerFactory
from ..sample_router import SampleRouter

class AlignerManager:
    def __init__(self, aligner_config):
        self.aligner_config = aligner_config
        self.routers = {}  # map of twig stream IDs to sample routers
        self.clear_keys = {}  # map of stream IDs to sets of clear keys
        self._aligner = None

    @property
    def aligner(self):
        if self._aligner:
            return self._aligner
        self._aligner = AlignerFactory().create_aligner(self.aligner_config)
        return self._aligner

    def add_router(self, stream_id, router_config):
        if stream_id in self.routers:
            self.remove_router(stream_id)
        self.routers[stream_id] = SampleRouter(router_config)
        self.clear_keys[stream_id] = set()

    def remove_router(self, stream_id):
        del self.routers[stream_id]
        # keep clear keys in case the router is added back later

    def get_aligner(self):
        return self.aligner

    def update_measurements(self, coord_sys_id, measurements):
        try:
            self.aligner.get_coordinate_system(coord_sys_id).update_measurements(measurements)
            return
        except ValueError as e:
            error_message = str(e)
            pass
        raise ValueError(f"Error updating measurements for coordinate system {coord_sys_id}: {error_message}")

    def process_twig(self, twig):
        stream_id = twig.stream_id
        coord_sys_id = twig.coord_sys_id

        if stream_id not in self.routers:
            raise ValueError(f"No router found for stream ID: {stream_id}")

        router = self.routers[stream_id]

        # Clear old measurements based on clear keys
        self._clear_measurements(stream_id, coord_sys_id)

        # Unpack new measurements
        new_measurements = router.unpack_measurements(twig)

        # Update the aligner with new measurements
        self.update_measurements(coord_sys_id, new_measurements)

        # Update clear keys
        self._update_clear_keys(stream_id, new_measurements)

    def _clear_measurements(self, stream_id, coord_sys_id):
        if stream_id in self.clear_keys:
            for clear_key in self.clear_keys[stream_id]:
                self.aligner.get_coordinate_system(coord_sys_id).clear_measurements(clear_key)

    def _update_clear_keys(self, stream_id, measurements):
        new_clear_keys = set()
        for measurement in measurements:
            new_clear_keys.add(measurement.clear_key)
        self.clear_keys[stream_id] = new_clear_keys
