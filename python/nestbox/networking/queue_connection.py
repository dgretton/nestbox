import queue
from typing import Tuple, Dict, Any
from nestbox.interfaces import ConnectionInterface

class QueueConnection(ConnectionInterface):
    def __init__(self, send_queue: queue.Queue, receive_queue: queue.Queue):
        self._send_queue = send_queue
        self._receive_queue = receive_queue
        self._connected = False
        self._connection_info = {}

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def send(self, data: bytes) -> None:
        if not self._connected:
            raise ConnectionError("Not connected")
        self._send_queue.put(data)

    def receive(self) -> bytes:
        if not self._connected:
            raise ConnectionError("Not connected")
        try:
            return self._receive_queue.get(block=False)
        except queue.Empty:
            return b''

    def is_connected(self) -> bool:
        return self._connected

    @property
    def connection_info(self) -> Dict[str, Any]:
        return self._connection_info

class QueueConnectionPair:
    def __init__(self):
        queue_a_to_b = queue.Queue()
        queue_b_to_a = queue.Queue()

        self.connection_a = QueueConnection(queue_a_to_b, queue_b_to_a)
        self.connection_b = QueueConnection(queue_b_to_a, queue_a_to_b)

    def create_connection_pair(self) -> Tuple[ConnectionInterface, ConnectionInterface]:
        return self.connection_a, self.connection_b
