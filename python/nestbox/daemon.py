from nestbox.interfaces import PeerDiscoveryInterface, ConnectionInterface, DatabaseInterface, AlignerClientInterface
from nestbox.daemon_local_aligner_client import DaemonLocalAlignerClient
import uuid
import time

class NestboxDaemon:
    _instance = None

    def __new__(cls): # Singleton pattern
        if cls._instance is None:
            cls._instance = super(NestboxDaemon, cls).__new__(cls)
        return cls._instance

    def initialize(self, config): # must be called before using the daemon
        self.config = config
        self.peer_discovery_client = self.initialize_peer_discovery_client()
        assert isinstance(self.peer_discovery_client, PeerDiscoveryInterface)
        self.database_client = self.initialize_database_client()
        assert isinstance(self.database_client, DatabaseInterface)
        self.aligner_client = self.initialize_aligner_client()
        assert isinstance(self.aligner_client, AlignerClientInterface)
        self.peer_connections = {}
        self.coordsys_names = CSNames()

    def initialize_peer_discovery_client(self):
        # Check if peer discovery process is running
        if not self.is_peer_discovery_client_running():
            self.launch_peer_discovery_client()
        return PeerDiscoveryClientImplementation(self.config['peer_discovery'])
    
    def is_peer_discovery_client_running(self):
        return True  # Placeholder

    def initialize_database_client(self):
        return DatabaseClientImplementation(self.config['database'])
    
    def is_database_client_running(self):
        return True

    def initialize_aligner_client(self):
        return DaemonLocalAlignerClient(self.config['aligner'])
    
    def is_aligner_client_running(self):
        return True

    def establish_peer_connections(self):
        peers = self.peer_discovery_client.get_peers()
        for peer in peers:
            if peer.id not in self.peer_connections:
                connection = self.create_connection(peer.connection_info)
                self.peer_connections[peer.id] = connection

    def create_coordsys(self):
        #new guid
        cs_guid = str(uuid.uuid4())
        #add to aligner
        blocker = [True, None]
        def unblock(guid):
            blocker[0] = False
            blocker[1] = guid
        self.aligner_client.create_coordinate_system(cs_guid, callback=unblock)
        print("Waiting for alignment client to create coordinate system")
        while blocker[0]:
            time.sleep(0.02)
        print("Coordinate system created, unblocked.")
        return blocker[1]

    def name_coordsys(self, cs_guid, cs_name):
        try:
            self.coordsys_names.set_name(cs_guid, cs_name)
        except ValueError as e:
            return False
        return True
    
    def add_measurements(self, cs_guid, measurements):
        blocker = [True, None]
        def unblock(success):
            blocker[0] = False
            blocker[1] = success
        self.aligner_client.add_measurements(cs_guid, measurements, callback=unblock)
        while blocker[0]:
            time.sleep(0.02)
        if not blocker[1]:
            raise RuntimeError("Failed to add measurements")

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


class PeerDiscoveryClientImplementation(PeerDiscoveryInterface):
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
    def get_feature_file(self):
        pass
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
