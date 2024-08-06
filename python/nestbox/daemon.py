from nestbox.interfaces import PeerDiscoveryClientInterface, ConnectionInterface, DatabaseInterface, AlignerClientInterface, PeerInfo
from nestbox.daemon_local_aligner_client import DaemonLocalAlignerClient
from nestbox.protos import Twig, MeasurementSet
from nestbox.numutil import coerce_numpy
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

class NestboxDaemon:
    _instance = None

    def __new__(cls): # Singleton pattern
        if cls._instance is None:
            cls._instance = super(NestboxDaemon, cls).__new__(cls)
        return cls._instance

    def initialize(self, config): # must be called before using the daemon
        self.config = config
        self.peer_discovery_client = self.initialize_peer_discovery_client()
        assert isinstance(self.peer_discovery_client, PeerDiscoveryClientInterface)
        self.database_client = self.initialize_database_client()
        assert isinstance(self.database_client, DatabaseInterface) 
        self.aligner_client = self.initialize_aligner_client()
        assert isinstance(self.aligner_client, AlignerClientInterface)
        self.peer_connections = {}
        self.coordsys_names = CSNames()
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor()

    def initialize_peer_discovery_client(self):
        # Check if peer discovery process is running
        if not self.is_peer_discovery_client_running():
            self.launch_peer_discovery_client()
        return PeerDiscoveryClientImplementation(self.config['peer_discovery']) # TODO: will be instantiated by a factory, explicit class will not be present
    
    def is_peer_discovery_client_running(self):
        return True  # Placeholder

    def initialize_database_client(self):
        return DatabaseClientImplementation(self.config['database']) # TODO: will be instantiated by a factory, explicit class will not be present
    
    def is_database_client_running(self):
        return True

    def initialize_aligner_client(self):
        return DaemonLocalAlignerClient(self.config['aligner']) # TODO: will be instantiated by a factory, explicit class will not be present
    
    def is_aligner_client_running(self):
        return True

    def establish_peer_connections(self):
        peers = self.peer_discovery_client.get_peers()
        for peer in peers:
            if peer.id not in self.peer_connections:
                connection = self.create_connection(peer.connection_info)
                self.peer_connections[peer.id] = connection

    def create_coordsys(self):
            print("Creating coordinate system")
            cs_guid = str(uuid.uuid4())
            print(f"Coordinate system GUID: {cs_guid}")
            future = self.executor.submit(self.aligner_client.create_coordinate_system, cs_guid)
            print("Coordinate system created")
            result = future.result()
            print(f"Result: {result}")
            return result

    def name_coordsys(self, cs_guid, cs_name):
        future = self.executor.submit(self.coordsys_names.set_name, cs_guid, cs_name)
        return future.result()

    def add_measurement(self, cs, measurement):
        cs_guid = self.resolve_cs_name(cs)
        return self.add_measurements(cs_guid, [measurement])
    
    def add_measurements(self, cs, measurements):
        cs_guid = self.resolve_cs_name(cs)
        future = self.executor.submit(self.aligner_client.add_measurements, cs_guid, measurements)
        return future.result()
    
    def start_aligner(self):
        future = self.executor.submit(self.aligner_client.start_alignment)
        return future.result()

    def get_basis_change_transform(self, source_cs, target_cs):
        print('Daemon: get_basis_change_transform called.')
        source_guid = self.resolve_cs_name(source_cs)
        target_guid = self.resolve_cs_name(target_cs)
        future = self.executor.submit(self.aligner_client.get_basis_change_transform, source_guid, target_guid)
        return future.result()['transform']

    def add_measurements_convoluted(self, cs, measurements):
        cs_guid = self.resolve_cs_name(cs)
        stream_id = str(uuid.uuid4())
        samples = []
        router_meas_configs = []
        dimensions = None
        is_homog = None
        for i, meas in enumerate(measurements):
            if meas['type'] == 'NormalMeasurement':
                given_dims = meas['dimensions']
                # if dimensions is already defined and the new measurement has different dimensions, raise an error
                if given_dims is not None:
                    if dimensions and not all(dim == given_dims[i] for i, dim in enumerate(dimensions)):
                        raise ValueError("All measurements added at the same time must have the same dimensions at the moment")
                    dimensions = given_dims
                # same for homogeneity
                given_homog = meas['is_homogeneous']
                if given_homog is not None:
                    if is_homog and not all(homog == given_homog[i] for i, homog in enumerate(is_homog)):
                        raise ValueError("All measurements added at the same time must have the same homogeneity at the moment")
                    is_homog = given_homog
                samples.append((coerce_numpy(meas['mean']), coerce_numpy(meas['covariance'])))
                router_meas_configs.append({
                    "type": meas['type'],
                    "feature_uri": meas['feature'],
                    "sample_pointer": {
                        "set": 0,
                        "sample": i
                    },
                    "clear_key": meas['feature']
                })
            else:
                raise ValueError(f"Unsupported measurement type: {meas['type']}")
        if dimensions is None:
            raise ValueError("No dimensions provided")
        if is_homog is None:
            raise ValueError("No is_homogenous flags provided")
        meas_set = MeasurementSet(samples, dimensions, is_homog)
        twig = Twig(stream_id, cs_guid, [meas_set])
        # add a router for a stream, then pack the measurements into a twig with that stream and send it to the aligner
        # Ssample routing configuration:
        # routing_info = {
        #     "stream_id": "test_stream",
        #     "measurements": [
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature1/position",
        #             "sample_pointer": {
        #                 "set": 0,
        #                 "sample": 0
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         },
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature2/position",
        #             "sample_pointer": {
        #                 "set": 0,
        #                 "sample": 1
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         },
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature3/position",
        #             "sample_pointer": {
        #                 "set": 0,
        #                 "sample": 2
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         },
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature4/position",
        #             "sample_pointer": {
        #                 "set": 1,
        #                 "sample": 0
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         },
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature5/position",
        #             "sample_pointer": {
        #                 "set": 1,
        #                 "sample": 1
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         },
        #         {
        #             "type": "normal",
        #             "feature_uri": "nestbox:feature/tag/features/point/feature6/position",
        #             "sample_pointer": {
        #                 "set": 1,
        #                 "sample": 2
        #             },
        #             "clear_key": "nestbox:feature/tag/features/point"
        #         }
        #     ]
        # }
        routing_info = {
            "stream_id": stream_id,
            "measurements": router_meas_configs
        }
        print('Daemon: routing info is:')
        print(routing_info)
        try:
            print(f'Daemon: twig bytes are {twig.to_bytes()}')
        except Exception as e:
            print(f"Error converting twig to bytes: {str(e)}")
        def add_router_and_process_twig():
            print(f"Daemon: adding router for stream {stream_id}")
            self.aligner_client.set_router(stream_id, routing_info)
            print(f"Daemon: added router for stream {stream_id}")
            self.aligner_client.send_twig(twig)
            print(f"Daemon: processed twig for stream {stream_id}")
        future = self.executor.submit(add_router_and_process_twig)
        return future.result()

    def handle_coordsys_request(self, cs_guids):
        # pseudocode
        # 1. Ask aligner for all the current coordinate system origins and orientations
        #    - aligner should always have the most recent results, so no need to check db or peers
        #    - in the future, we can see if peers have a more certain alignment vs a more recent one & trade that off,
        #      probably via the coordinate system property that specifies how long the alignment is expected
        #      to be valid (sort of an expiration time)
        #    - note for the future, the PURSE radius can quantify alignment certainty in SE(3)
        # 2. For any that aren't found, ask peers for alignment results and also check if they are available in the database
        cs_alignments = {guid:None for guid in cs_guids}
        self.aligner_client.get_cs_status(cs_guids)
        if not alignment_data:
            for peer_id, connection in self.peer_connections.items():
                peer_data = self.request_alignment_data(connection, cs_guids)
                if peer_data:
                    alignment_data.update(peer_data)
        alignment_data = self.database_client.get_alignment_data(cs_guids)

    def request_alignment_data(self, connection, cs_guids):
        # Use the PeerDiscoveryAPI to request data from a peer
        pass

    def resolve_cs_name(self, cs):
        try:
            cs_guid = self.coordsys_names.get_guid(cs)
            if cs_guid is not None:
                return cs_guid
        except KeyError:
            pass
        return cs

    def process_request(self, request):
        # Example request processing logic
        print(f"Received request: {request}")
        return f"Processed: {request}"  # Simple echo for demonstration
    
    def start(self):
        if not self.is_database_client_running():
            self.database_client.connect()
        # while we're using the daemon-local alignment "client" (really a client interface thinly wrapping an aligner that it itself launches, running in a thread) no need to start the aligner, it's initialized and you can send a start_alignment request to it
        # TODO: kick off a peer discovery process with a callback to connect to all
        # the returned connections, starting with something like 
        # self.peer_discovery_client.start_discovery()


class PeerInfoImplementation(PeerInfo):
    pass


class PeerDiscoveryClientImplementation(PeerDiscoveryClientInterface):
    def get_peers(self):
        # For now, return a static list of peer info
        return [ConnectionConfig('localhost', 12345, 'tcp', 'peer1'), ConnectionConfig('localhost', 12346, 'tcp', 'peer2')]
    def connect(self):
        pass
    def connection(self):
        pass
    def connection_info(self):
        pass
    def disconnect(self):
        pass
    def is_connected(self):
        return True
    def on_peer_discovered(self):
        pass
    def on_peer_lost(self):
        pass
    def register_self(self):
        pass
    def start_discovery(self):
        pass
    def stop_discovery(self):
        pass
    def unregister_self(self):
        pass


class DatabaseClientImplementation(DatabaseInterface):
    def __init__(self, config):
        self.config = config
    def get_alignment_data(self, cs_guids):
        # Retrieve alignment data from the database
        pass
    def begin_transaction(self):
        pass
    def commit_transaction(self):
        pass
    def connect(self):
        pass
    def disconnect(self):
        pass
    def execute_query(self):
        pass
    def get_alignment_history(self):
        pass
    def get_alignment_result(self):
        pass
    def get_feature_file(self, *args): # TODO: placeholder
        return (
        """<feature-definition xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                    xsi:noNamespaceSchemaLocation="feature_schema.xsd">
        <feature-type name="Hand" multiple="true">
            <description>Hand tracking feature</description>
            <root-pose>
            <position dimensions="XYZ">
                <allowed-measurements>
                <measurement-type>NormalMeasurement</measurement-type>
                <measurement-type>OptionsMeasurement</measurement-type>
                </allowed-measurements>
                <default-measurement>NormalMeasurement</default-measurement>
            </position>
            <orientation dimensions="IJK">
                <allowed-measurements>
                <measurement-type>NormalMeasurement</measurement-type>
                </allowed-measurements>
                <default-measurement>NormalMeasurement</default-measurement>
            </orientation>
            </root-pose>
            <features>
            <feature-type name="TrackingPoint" multiple="true">
                <position dimensions="XYZ">
                <allowed-measurements>
                    <measurement-type>NormalMeasurement</measurement-type>
                    <measurement-type>CollectionMeasurement</measurement-type>
                </allowed-measurements>
                <default-measurement>NormalMeasurement</default-measurement>
                </position>
            </feature-type>
            </features>
        </feature-type>
        </feature-definition>"""
        )
    def get_feature_file_metadata(self):
        pass
    def get_measurement(self):
        pass
    def get_measurements(self):
        pass
    def rollback_transaction(self):
        pass
    def store_alignment_result(self):
        pass
    def store_feature_file(self):
        pass
    def store_measurement(self):
        pass


class CSNames:
    def __init__(self):
        self.names = {}
        self.cs_guids = {}

    def get_name(self, cs_guid):
        return self.names.get(cs_guid)
    
    def get_guid(self, cs_name):
        return self.cs_guids.get(cs_name)
    
    def set_name(self, cs_guid, cs_name):
        if cs_name in self.cs_guids and self.cs_guids[cs_name] != cs_guid:
            raise ValueError(f"Name '{cs_name}' already exists for a different coordinate system, GUID {self.cs_guids[cs_name]}")
        self.names[cs_guid] = cs_name
        self.cs_guids[cs_name] = cs_guid

    def set_guid(self, cs_name, cs_guid):
        self.cs_guids[cs_name] = cs_guid
        self.names[cs_guid] = cs_name

    def remove_name(self, cs_guid):
        cs_name = self.names.pop(cs_guid)
        self.cs_guids.pop(cs_name)
        return cs_name
    
    def remove_guid(self, cs_name):
        cs_guid = self.cs_guids.pop(cs_name)
        self.names.pop(cs_guid)
        return cs_guid

global_daemon = NestboxDaemon()
