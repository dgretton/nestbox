import threading
import queue
import json
import uuid
import base64
import time
from typing import Dict, Any, List
from nestbox.interfaces import AlignerClientInterface, ConnectionInterface, AlignmentResultInterface, ServerInterface, ServerConnectionInterface
from nestbox.networking import QueueConnectionPair
from nestbox.aligner import AlignerManager
from nestbox.measurement import NormalMeasurement
from nestbox.feature import StrFeatureKey
from nestbox.coordsystem import CoordinateSystem
from nestbox.run_optimizer import run_optimizer
from nestbox.protos import Twig
from nestbox.visualizer import Visualizer

class AsyncResponseHandler:
    def __init__(self, connection):
        self.pending_requests = {}
        assert isinstance(connection, ConnectionInterface)
        self._connection = connection
        self.callbacks = {}

    def generate_request_id(self):
        return str(uuid.uuid4())
    
    def start(self):
        print("Starting async message handler")
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

    def _receive_loop(self):
        print("Starting async message handler receive loop")
        if not self._connection.is_connected():
            self._connection.connect()
        try:
            while self.running:
                try:
                    data = self._connection.receive()
                    if data:
                        message = json.loads(data.decode())
                        self.handle_response(message)
                except ConnectionError:
                    self.running = False
                    break
        finally:
            self._connection.disconnect()

    def stop(self):
        self.running = False

    def send_request(self, request_type, request_data, callback=None):
        print(f"Sending async message handler request: {request_type}")
        request_id = self.generate_request_id()
        request = {
            "type": request_type,
            "request_id": request_id,
            **request_data
        }
        self.pending_requests[request_id] = request
        if callback is not None:
            self.callbacks[request_id] = callback
        print(f"really right about to send async message handler request: {request}")
        if not self._connection.is_connected():
            self._connection.connect()
        self._connection.send(json.dumps(request).encode())
        print(f"Sent async message handler request: {request}")
        return request_id

    def handle_response(self, response):
        request_id = response.get("request_id")
        #check type equals response
        if response.get("type") != "response":
            raise ValueError("Received message is not a response")
        if request_id in self.pending_requests:
            original_request = self.pending_requests.pop(request_id)
            callback = self.callbacks.pop(request_id, None)
            self._handle_response(original_request, response, callback)
        else:
            raise ValueError(f"Received response for unknown request ID {request_id}")

    def _handle_response(self, request, response, callback):
        raise NotImplementedError("Subclasses must implement _handle_response()")


class _DaemonLocalAlignerServer(ServerInterface):
    def __init__(self, server_connection):
        assert isinstance(server_connection, ServerConnectionInterface)
        self._server_connection = server_connection
        self.aligner_manager = AlignerManager({'type': 'adam', 'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'temperature': 20})
        #self.aligner_manager = AlignerManager({'type': 'gradient', 'learning_rate': 0.01})
        self.aligner_thread = threading.Thread(target=self._run_aligner, daemon=True)
        #self.aligner_thread.start() #TODO: remove, should wait until receiving aligner start request
    
    @property
    def aligner(self):
        return self.aligner_manager.get_aligner()
    
    def _run_aligner(self):
        # Connect to Redis
        import redis
        redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        # check if connected
        try:
            redis_client.ping()
        except redis.exceptions.ConnectionError:
            print("Redis server not running. Start Redis server with 'redis-server'.")

        # Function to publish optimization updates
        def publish_updates(channel, state):
            # state: a dictionary containing the current state of the optimization
            try:
                redis_client.publish(channel, json.dumps(state))
            except redis.exceptions.ConnectionError:
                pass

        def redis_listener():
            pubsub = redis_client.pubsub()
            pubsub.subscribe(['optimization_update', 'pin_command'])
            print("Redis listener started")

            for message in pubsub.listen():
                if message['type'] == 'message':
                    # if message['channel'].decode('utf-8') == 'optimization_update':
                    #     data = json.loads(message['data'])
                    #     # Handle optimization update
                    if message['channel'].decode('utf-8') == 'pin_command':
                        pin_data = json.loads(message['data'])
                        pin_index = pin_data['pin']
                        print(f"Received pin command for coordinate system {pin_index}")
                        self.aligner.pin(pin_index)

        # Start the listener in a separate thread
        threading.Thread(target=redis_listener, daemon=True).start()

        # Visualizer
        visualizer = Visualizer(self.aligner)

        def callback(aligner):
            for _, origin, orientation in aligner.iterate_coordinate_systems():
                    print(_.name)
                    print(f"    current coordinate system position: {origin}")
                    print(f"    current coordinate system orientation: {orientation}")
            # Send optimization state to Redis
            visualizer.draw()
            state = visualizer.state()
            publish_updates('optimization_update', state)
        
        run_optimizer(self.aligner, callback=callback)

    def start(self):
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    def _serve(self):
        self._server_connection.connect()
        print(f"Server listening")
        while True:
            client_connection = self._server_connection.accept()
            client_thread = threading.Thread(target=self.handle_connection, args=(client_connection,))
            client_thread.start()

    def stop(self):
        self._server_connection.disconnect()

    def handle_connection(self, connection):
        if not connection.is_connected():
            connection.connect()
        while self._server_connection.is_connected():
            print(f"receiving data from connection: {connection.connection_info}")
            request_data = connection.receive()
            print("data received")
            if request_data:
                # Process data
                response = self.process_data(request_data)
                connection.send(response)
            else:
                connection.disconnect()

    def process_data(self, request_data: bytes) -> bytes:
        print(f"Received request: {request_data}")
        request = json.loads(request_data.decode())
        print(f"JSON request: {request}")
        request_type = request['type']
        response = {
            "type": "response",
            "request_type": request_type,
            "request_id": request['request_id'],
            "status": "error"
        }

        def success():
            response.update({"status": "success"})

        print(f"Processing request type: {request_type}")
        if request_type == 'create_cs':
            cs_guid = request['cs_guid']
            coord_sys = CoordinateSystem(name=cs_guid)
            self.aligner.add_coordinate_system(coord_sys)
            print('HOLY SH*T WE ACTUALLY INITIALIZED A COORDINATE SYSTEM ALL THE WAAAAY')
            success()
        elif request_type == 'add_measurements':
            cs_guid = request['cs_guid']
            measurements_data = request['measurements']
            measurements = []
            for meas_data in measurements_data:
                if meas_data['type'] == 'NormalMeasurement':
                    feature = StrFeatureKey(meas_data['feature'])
                    mean = meas_data['mean']
                    covariance = meas_data['covariance']
                    dimensions = meas_data.get('dimensions')
                    is_homogeneous = meas_data.get('is_homogeneous')
                    measurements.append(NormalMeasurement(feature, mean, covariance, dimensions, clear_key=None))
                else:
                    raise ValueError(f"Unsupported measurement type: {meas_data['type']}")
            try:
                self.aligner_manager.update_measurements(cs_guid, measurements)
                success()
            except ValueError as e:
                response.update({"status": "error", "message": str(e)})
        elif request_type == 'add_twig':
            #data in bytes is base64 encoded in field twig_data
            data64 = request['twig_data']
            data = base64.b64decode(data64)
            twig = Twig().load_bytes(data)
            print(f"Received Twig: {twig}")
            self.aligner_manager.process_twig(twig)
            success()
        elif request_type == 'set_router':
            stream_id = request['stream_id']
            router_config = request['router_config']
            self.aligner_manager.add_router(stream_id, router_config)
            success()
        elif request_type == 'start_alignment':
            if not self.aligner_thread.is_alive():
                print("Starting aligner thread")
                self.aligner_thread.start()
            success()
        elif request_type == 'get_basis_change_transform':
            source_cs = request['source_cs']
            target_cs = request['target_cs']
            print(f'LocalAlignerClient: getting basis change transform from {source_cs} to {target_cs}')
            transform = self.aligner.get_basis_change_transform(source_cs, target_cs)
            print(f'Successfully got basis change transform:')
            print(transform)
            print(transform.to_json())
            response.update({"status": "success", "transform": transform.to_json()})
        elif request_type == 'cancel_alignment':
            pass
        elif request_type == 'alignment_status':
            pass
        elif request_type == 'get_cs_status':
            pass
        elif request_type == 'get_all_cs':
            pass
        elif request_type == 'get_latest_alignments':
            pass
        elif request_type == 'set_alignment_params':
            pass
        elif request_type == 'delete_cs':
            pass
        elif request_type == 'pin':
            pass
        elif request_type == 'unpin':
            pass
        print(f"Sending response from process_data: {response}")
        return json.dumps(response).encode()


class SingleServerConnection(ServerConnectionInterface):
    def __init__(self, single_connection):
        self._single_connection = single_connection
        self.accepted = False

    def accept(self) -> ConnectionInterface:
        if not self.accepted:
            self.accepted = True
            return self._single_connection
        time.sleep(365 * 24 * 60 * 60)  # Sleep for a year

    def connect(self):
        pass

    def disconnect(self):
        pass

    def is_connected(self):
        return True
    
    @property
    def connection_info(self):
        return {"type": "single_connection"}

class DaemonLocalAsyncResponseHandler(AsyncResponseHandler):
    def _handle_response(self, request, response, callback):
        #TODO: custom handling & callback-ing for different request types
        if callback is None:
            return
        if request['type'] == 'create_cs':
            if response['status'] == 'success':
                callback(request['cs_guid'])
            else:
                callback(None)
            return
        try:
            callback(response)
        except TypeError:
            callback()

class DaemonLocalAlignerClient(AlignerClientInterface):
    def __init__(self, config):
        self._connection, aligner_server_conn = QueueConnectionPair().get_connection_pair()
        server_conn = SingleServerConnection(aligner_server_conn)
        server = _DaemonLocalAlignerServer(server_conn)
        server.start()
        self._connection.connect()
        self.handler = DaemonLocalAsyncResponseHandler(self._connection)
        self.handler.start() # start the handler's receive loop (it automatically runs in a new thread)
        
    def start_alignment(self) -> None:
        self.handler.send_request("start_alignment", {}, callback=lambda: print("Alignment started successfully"))

    def cancel_alignment(self) -> None:
        ({"type": "cancel_alignment"})
        

    def get_alignment_status(self) -> Dict[str, Any]:
        ({"type": "alignment_status"})
        

    def get_cs_status(self, cs_guids: List[str]) -> Dict[str, Any]:
        ({"type": "get_cs_status", "cs_guids": cs_guids})
        

    def get_coordinate_systems(self) -> Dict[str, Any]:
        ({"type": "get_all_cs"})
        

    def get_latest_alignments(self, cs_guids: List[str]) -> Dict[str, AlignmentResultInterface]:
        ({"type": "get_latest_alignments", "cs_guids": cs_guids})
        
    def get_basis_change_transform(self, source_cs_guid: str, target_cs_guid: str) -> Dict[str, Any]:
        print('LocalAlignerClient: get_basis_change_transform called, sending "get_basis_change_transform" query to aligner\n'
              f'source guid: {source_cs_guid}, target guid: {target_cs_guid}')
        return self._wait_for_callback_result('get_basis_change_transform', {"source_cs": source_cs_guid, "target_cs": target_cs_guid})

    def set_alignment_parameters(self, params: Dict[str, Any]) -> None:
        ({"type": "set_alignment_params", "params": params})

    def create_coordinate_system(self, guid: str) -> str:
        return self._wait_for_callback_result('create_cs', {"cs_guid": guid})

    def delete_coordinate_system(self, cs_guid: str) -> str:
        ({"type": "delete_cs", "cs_guid": cs_guid})

    def add_measurements(self, cs_guid: str, measurements: List[Dict[str, Any]]) -> str:
        return self._wait_for_callback_result('add_measurements', {"cs_guid": cs_guid, "measurements": measurements})

    def set_router(self, stream_id: str, router_config: Dict[str, Any]) -> str:
        return self._wait_for_callback_result('set_router', {"stream_id": stream_id, "router_config": router_config})
    
    def send_twig(self, twig: Twig) -> str:
        twig_data = twig.to_bytes()
        print(f"Serialized twig data in send_twig: {twig_data}")
        return self._wait_for_callback_result('add_twig', {"twig_data": base64.b64encode(twig_data).decode()})

    def pin_coordinate_system(self, cs_guid: str) -> str:
        ({"type": "pin", "cs_guid": cs_guid})

    def unpin_coordinate_system(self) -> str:
        ({"type": "unpin"})

    def _wait_for_callback_result(self, *args, **kwargs):
        result = {'value': None}
        event = threading.Event()

        def callback(response):
            result['value'] = response
            event.set()

        self.handler.send_request(*args, **kwargs, callback=callback)
        event.wait()
        return result['value']

    # Implement ClientInterface methods
    def connect(self, connection):
        pass  # Not needed for this dummy implementation

    def disconnect(self):
        pass  # Not needed for this dummy implementation

    def is_connected(self):
        return True  # Always return True for this dummy implementation

    @property
    def connection(self):
        return self._connection

    @property
    def connection_info(self):
        return self._connection.connection_info

    # Implement PubSubInterface methods
    def publish(self, topic: str, message: Any) -> None:
        pass  # Not implemented in this dummy version

    def subscribe(self, topic: str, callback) -> None:
        pass  # Not implemented in this dummy version

    def register_callback(self, topic: str, callback) -> None:
        pass  # Not implemented in this dummy version

    def unregister_callback(self, topic: str, callback) -> None:
        pass  # Not implemented in this dummy version

    def unsubscribe(self, topic: str) -> None:
        pass  # Not implemented in this dummy version


class DummyAlignmentResult(AlignmentResultInterface):
    def __init__(self, cs_guid):
        self._timestamp = time.time()
        self._status = "completed"
        self._matrix_transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._origin = [0, 0, 0]
        self._quaternion = [1, 0, 0, 0]

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def status(self) -> str:
        return self._status

    @property
    def matrix_transform(self) -> List[List[float]]:
        return self._matrix_transform

    @property
    def origin(self) -> List[float]:
        return self._origin

    @property
    def quaternion(self) -> List[float]:
        return self._quaternion

# The following schematic JSON messages are expected to be sent and received by the SimpleAlignerClient class:
# {
#     "request_id":"(uuid4)",
#     "type": "create_cs",
#     "cs_guid": "unique-guid-string"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "add_twig",
#     "twig_data": "base64-encoded-twig-data"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "add_measurement_set",
#     "coord_sys_id": "cs-guid-12345",
#     "stream_id": "stream-guid-67890",
#     "measurements": [
#     {
#       "dimensions": ["X", "Y", "Z"],
#       "is_homogeneous": [false, false, false],
#       "samples": [
#         {
#           "mean": [1.0, 2.0, 3.0],
#           "covariance": {
#             "upper_triangle": [0.01, 0.002, 0.03, 0.003, 0.004, 0.05]
#           }
#         },
#         {
#           "mean": [1.5, 2.5, 3.5],
#           "covariance": {
#             "upper_triangle": [0.02, 0.003, 0.04, 0.004, 0.005, 0.06]
#           }
#         }
#       ],
#       "transform": {
#         "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#       }
#     },
#     {
#       "dimensions": ["X", "Y", "Z", "T"],
#       "is_homogeneous": [true, true, true, false],
#       "samples": [
#         {
#           "mean": [4.0, 5.0, 6.0, 1.5],
#           "covariance": {
#             "upper_triangle": [0.01, 0.002, 0.03, 0.004, 0.003, 0.04, 0.005, 0.006, 0.07, 0.01]
#           }
#         }
#       ],
#       "transform": {
#         "data": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#       }
#     }
#     ]
# }
# {
#     "request_id":"(uuid4)",
#     "type": "start_alignment"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "cancel_alignment"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "alignment_status"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "get_cs_status",
#     "cs_guids": ["guid1", "guid2", "..."]
# }
# {
#     "request_id":"(uuid4)",
#     "type": "get_all_cs"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "get_latest_alignments",
#     "cs_guids": ["guid1", "guid2", "..."]
# }
# {
#     "request_id":"(uuid4)",
#     "type": "set_alignment_params",
#     "params": {
#         "learning_rate": 0.01
#     }
# }
# {
#     "request_id":"(uuid4)",
#     "type": "delete_cs",
#     "cs_guid": "coordinate-system-guid"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "pin",
#     "cs_guid": "coordinate-system-guid"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "unpin"
# }
# {
#     "request_id":"(uuid4)",
#     "type": "response",
#     "request_type": "get_alignment_status",
#     "status": "success",
#     "data": {
#         "status": "in_progress"
#     }
# }