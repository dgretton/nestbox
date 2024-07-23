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
        print("QueueConnection.receive()")
        if not self._connected:
            raise ConnectionError("Not connected")
        try:
            return self._receive_queue.get(block=True)
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

    def get_connection_pair(self) -> Tuple[ConnectionInterface, ConnectionInterface]:
        return self.connection_a, self.connection_b

def test_queue_connection_pair():
    import threading
    pair = QueueConnectionPair()
    conn_a, conn_b = pair.get_connection_pair()

    def send_data_a():
        print("started thread send_data_a()")
        try:
            conn_a.connect()
            conn_a.send(b"Hello, World!")
            print("Sent data")
        finally:
            conn_a.disconnect()
    
    threading.Thread(target=send_data_a, daemon=True).start()
    try:
        conn_b.connect()
        data = conn_b.receive()
        print(f"Received data: {data}")
    finally:
        conn_b.disconnect()

    def send_data_b():
        print("started thread send_data_b()")
        try:
            conn_b.connect()
            conn_b.send(b"Hello, World!")
            print("Sent data")
        finally:
            conn_b.disconnect()
    
    threading.Thread(target=send_data_b, daemon=True).start()
    try:
        conn_a.connect()
        data = conn_a.receive()
        print(f"Received data: {data}")
    finally:
        conn_a.disconnect()


if __name__ == "__main__":
    test_queue_connection_pair()
