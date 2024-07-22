import threading
import queue
import json
import uuid
import time
from typing import Dict, Any, List
from nestbox.interfaces import AlignerClientInterface, ConnectionInterface, AlignmentResultInterface
from nestbox.networking import QueueConnectionPair
from nestbox.aligner import AdamAligner
from nestbox.coordsystem import CoordinateSystem
from nestbox.run_optimizer import run_optimizer

class AsyncMessageHandler:
    def __init__(self, connection):
        self.pending_requests = {}
        assert isinstance(connection, ConnectionInterface)
        self._connection = connection
        self.callbacks = {}

    def generate_request_id(self):
        return str(uuid.uuid4())
    
    def start(self):
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

    def _receive_loop(self):
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
        request_id = self.generate_request_id()
        request = {
            "type": request_type,
            "request_id": request_id,
            **request_data
        }
        self.pending_requests[request_id] = request
        if callback is not None:
            self.callbacks[request_id] = callback
        self._connection.send(json.dumps(request).encode())
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


class AlignerHandler(AsyncMessageHandler):
    def __init__(self, connection, aligner):
        super().__init__(connection)
        self.aligner = aligner

    def _handle_response(self, request, response, callback):
        request_type = request['type']
        if callback is None:
            def callback(*args, **kwargs):
                pass
        if request_type == 'create_cs':
            cs_guid = response['data']['cs_guid']
        elif request_type == 'add_twig':
            pass
        elif request_type == 'add_measurement_set':
            pass
        elif request_type == 'start_alignment':
            pass
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


class DaemonLocalAlignerClient(AlignerClientInterface):
    def __init__(self, config):
        self.aligner = AdamAligner()
        self._connection, aligner_side_connection = QueueConnectionPair().create_connection_pair()
        self.handler = AlignerHandler(aligner_side_connection, self.aligner)
        threading.Thread(target=self._run_aligner, daemon=True).start()

    def _run_aligner(self):
        run_optimizer(self.aligner)

    def _send_request(self, request_type, request_data):
        pass
        
    def start_alignment(self) -> None:
        ({"type": "start_alignment"})


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
        

    def set_alignment_parameters(self, params: Dict[str, Any]) -> None:
        ({"type": "set_alignment_params", "params": params})
        

    def create_coordinate_system(self, guid: str) -> str:
        ({"type": "create_cs", "guid": guid})
        

    def delete_coordinate_system(self, cs_guid: str) -> str:
        ({"type": "delete_cs", "cs_guid": cs_guid})
        

    def pin_coordinate_system(self, cs_guid: str) -> str:
        ({"type": "pin", "cs_guid": cs_guid})
        

    def unpin_coordinate_system(self) -> str:
        ({"type": "unpin"})
        

    def get_response(self, request_id: str, timeout: float = None) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            try:
                response = self.result_queue.get(timeout=0.1)
                if response['request_id'] == request_id:
                    return response
                else:
                    # Put the response back if it's not the one we're looking for
                    self.result_queue.put(response)
            except queue.Empty:
                if timeout is not None and time.time() - start_time > timeout:
                    raise TimeoutError(f"No response received for request {request_id} within {timeout} seconds")

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
        self._alignment_id = f"alignment_{cs_guid}"
        self._timestamp = time.time()
        self._status = "completed"
        self._matrix_transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._origin = [0, 0, 0]
        self._quaternion = [1, 0, 0, 0]

    @property
    def alignment_id(self) -> str:
        return self._alignment_id

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
#     "guid": "unique-guid-string"
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