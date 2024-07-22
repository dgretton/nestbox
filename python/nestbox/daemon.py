from nestbox.networking import TCPDaemonServer, UnixSocketDaemonServer
from nestbox.interfaces import PeerDiscoveryInterface, ConnectionInterface, DatabaseInterface, AlignerClientInterface
from nestbox.daemon_local_aligner_client import DaemonLocalAlignerClient
import uuid

class NestboxDaemon:
    def __init__(self, config):
        self.config = config
        self.peer_discovery_client = self.initialize_peer_discovery_client()
        assert isinstance(self.peer_discovery_client, PeerDiscoveryInterface)
        self.database_client = self.initialize_database_client()
        assert isinstance(self.database_client, DatabaseInterface)
        self.aligner_client = self.initialize_aligner_client()
        assert isinstance(self.aligner_client, AlignerClientInterface)
        # Initialize the server based on configuration
        dconfig = config['daemon']
        server_type = dconfig['server_type']
        server_address = dconfig['server_address']
        if server_type.lower() == 'unix_socket':
            self.server = UnixSocketDaemonServer(server_address, self)
        elif server_type.lower() == 'tcp':
            self.server = TCPDaemonServer(server_address, self)
        else:
            raise ValueError(f"Invalid server type: {server_type}")
        self.connections = {}

    def initialize_peer_discovery_client(self):
        # Check if peer discovery process is running
        if not self.is_peer_discovery_client_running():
            self.launch_peer_discovery_client()
        return PeerDiscoveryClientImplementation(self.config['peer_discovery'])
    
    def is_peer_discovery_client_running(self):
        return True  # Placeholder

    def initialize_database_client(self):
        if not self.is_database_client_running():
            self.launch_database_client()
        return DatabaseClientImplementation(self.config['database'])
    
    def is_database_client_running(self):
        return True

    def initialize_aligner_client(self):
        if not self.is_aligner_client_running():
            self.launch_aligner_client()
        return DaemonLocalAlignerClient(self.config['aligner'])
    
    def is_aligner_client_running(self):
        return True

    def establish_peer_connections(self):
        peers = self.peer_discovery_client.get_peers()
        for peer in peers:
            if peer.id not in self.connections:
                connection = self.create_connection(peer.connection_info)
                self.connections[peer.id] = connection

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
            for peer_id, connection in self.connections.items():
                peer_data = self.request_alignment_data(connection, cs_guids)
                if peer_data:
                    alignment_data.update(peer_data)
        alignment_data = self.database_client.get_alignment_data(cs_guids)
        
        if alignment_data:
            return self.aligner_client.perform_alignment(alignment_data)
        else:
            return None

    def request_alignment_data(self, connection, cs_guids):
        # Use the PeerDiscoveryAPI to request data from a peer
        pass

    def process_request(self, request):
        # Example request processing logic
        print(f"Received request: {request}")
        return f"Processed: {request}"  # Simple echo for demonstration

    def start(self):
        try:
            self.server.start()
        except Exception as e:
            print(f"Failed to start the server: {e}")

    def stop(self):
        try:
            self.server.stop()
        except Exception as e:
            print(f"Failed to stop the server: {e}")


class PeerDiscoveryClientImplementation(PeerDiscoveryInterface):
    def get_peers(self):
        # For now, return a static list of peer info
        return [PeerInfo("peer1", ConnectionInfo(...), {...}),
                PeerInfo("peer2", ConnectionInfo(...), {...})]
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


if __name__ == "__main__":
    import yaml
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, help="Path to the configuration file")
    args = ap.parse_args()
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    daemon = NestboxDaemon(config)
    daemon.start()
