#TODO: move to top of module, not in networking submodule
from nestbox.interfaces import ServerInterface, ServerConnectionInterface
from nestbox.daemon import global_daemon
from nestbox.networking import ConnectionManager, ConnectionConfig
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class MessageType(Enum):
    CREATE_CS = "create_cs"
    NAME_CS = "name_cs"

@dataclass
class CreateCSResponse:
    cs_guid: str

@dataclass
class NameCSRequest:
    cs_guid: str
    name: str

def validate_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not request.is_json:
                return jsonify({"error": "Missing JSON in request"}), 400
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return decorated_function

class CreateCoordSysResource(Resource):
    @validate_json
    def post(self):
        print("Received create CS POST request")
        cs_guid = global_daemon.create_coordsys()
        print(f"Created CS with GUID: {cs_guid}")
        if not cs_guid:
            return {"error": "Failed to create coordinate system"}, 500
        return asdict(CreateCSResponse(cs_guid=cs_guid)), 201

class NameCoordSysResource(Resource):
    @validate_json
    def put(self, cs_guid):
        print(f"Received name CS PUT request for GUID: {cs_guid}")
        data = request.get_json()
        
        if 'name' not in data:
            return {"error": "Missing 'name' in request body"}, 400
        
        name = data['name']
        try:
            global_daemon.name_coordsys(cs_guid, name)
            return {"message": "Coordinate system named successfully"}, 200
        except Exception as e:
            print(f"Error naming coordinate system: {str(e)}")
            return {"error": "Failed to name coordinate system"}, 500
        
class AddMeasurementsResource(Resource):
    @validate_json
    def post(self, cs_guid):
        data = request.get_json()
        measurements = data.get('measurements')
        
        if not measurements:
            return {"error": "Missing 'measurements' in request body"}, 400
        
        try:
            global_daemon.add_measurements(cs_guid, measurements)
            return {"message": "Measurements added successfully"}, 200
        except Exception as e:
            print(f"Error adding measurements: {str(e)}")
            return {"error": "Failed to add measurements"}, 500

api.add_resource(CreateCoordSysResource, '/coordsys')
api.add_resource(NameCoordSysResource, '/coordsys/<string:cs_guid>/name')
api.add_resource(AddMeasurementsResource, '/coordsys/<string:cs_guid>/measurements')

# class DaemonServer(ServerInterface):
#     def __init__(self, config, daemon):
#         self.config = config
#         self.daemon = daemon
#         self.connection_manager = ConnectionManager()
#         self.server_connection = self.create_server_connection()

#     def create_server_connection(self):
#         dconfig = self.config['daemon']
#         server_type = dconfig['server_type']
#         server_address = dconfig['server_address']
#         if server_type == 'unix_socket':
#             connection_config = ConnectionConfig(server_type, address=server_address)
#         elif server_type == 'tcp':
#             port = int(dconfig['server_port'])
#             connection_config = ConnectionConfig(server_type, address=server_address, port=port)
#         else:
#             raise ValueError(f"Unsupported server type: {server_type}")
#         server_connection = self.connection_manager.create_server_connection(connection_config)
#         assert isinstance(server_connection, ServerConnectionInterface)
#         return server_connection

#     def start(self):
#         self.server_connection.connect()
#         print(f"Server listening")
#         while True:
#             client_connection = self.server_connection.accept()
#             client_thread = threading.Thread(target=self.handle_connection, args=(client_connection,))
#             client_thread.start()

#     def handle_connection(self, client_connection):
#         while client_connection.is_connected():
#             data = client_connection.receive()
#             if data:
#                 # Process data
#                 response = self.process_data(data)
#                 client_connection.send(response)
#             else:
#                 client_connection.disconnect()

#     def process_data(self, data: bytes) -> None:
#         import json
#         try:
#             message = json.loads(data.decode())
#             if message['type'] == 'create_cs':
#                 guid = self.daemon.handle_create_coordsys_request()
#                 if not guid:
#                     return json.dumps({'type': 'response', 'request_type': 'create_cs', 'status': 'error', 'message': 'Failed to create coordsys'}).encode()
#                 return json.dumps({'type': 'response', 'request_type': 'create_cs', 'guid': guid}).encode()
#             elif message['type'] == 'name_cs':
#                 cs_guid = message['guid']
#                 cs_name = message['name']
#                 self.daemon.handle_name_coordsys_request(cs_guid, cs_name)
#             else:
#                 raise ValueError(f"Unsupported message type: {message['type']}")
#         except json.JSONDecodeError:
#             print("Invalid JSON data received")
#         except KeyError:
#             print("Missing 'type' field in JSON data")
#         except ValueError as e:
#             print(e)

#     def stop(self):
#         self.server_connection.disconnect()

def run_daemon():
    global_daemon.start()

daemon_thread = threading.Thread(target=run_daemon)

@app.before_first_request
def start_daemon():
    daemon_thread.start()

if __name__ == '__main__':
    import socket
    import os
    import signal
    import yaml
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, help="Path to the configuration file")
    args = ap.parse_args()
    
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    global_daemon.initialize(config)
    
    from nestbox.config import DAEMON_CONN_CONFIG as conn
    
    if conn.type == 'unix_socket':
        socket_path = conn.address
        
        # Check if the directory for the socket exists
        socket_dir = os.path.dirname(socket_path)
        if not os.path.exists(socket_dir):
            raise FileNotFoundError(f"Directory for Unix socket does not exist: {socket_dir}")

        # Remove the socket file if it already exists
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("Shutting down...")
            os.unlink(socket_path)
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print(f"Starting server at unix:{socket_path}")
        
        # Run the Flask app with the Unix socket
        app.run(host=f'unix://{socket_path}', debug=True, use_reloader=False)
    
    else:
        if conn.type == 'tcp':
            host = conn.address
            port = conn.port
        else:  # Assume HTTP
            host = conn.address
            port = 80  # Default HTTP port, adjust if needed
        
        print(f"Starting server at {host}:{port}")
        app.run(host=host, port=port, debug=True, use_reloader=False)
