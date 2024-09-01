import socket
import threading
from nestbox.interfaces import ConnectionInterface
from nestbox.protos import Twig

class LiveDataShim:
    def __init__(self, connection: ConnectionInterface):
        from nestbox.daemon import global_daemon
        self.global_daemon = global_daemon
        self.host = connection.connection_info['host'] # for simplicity for this patch just reconstruct the connection from the connection info
        self.port = connection.connection_info['port']
        self.server_socket = None
        self.running = False
        self.firsttime = True

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        threading.Thread(target=self._accept_connections, daemon=True).start()

    def _accept_connections(self):
        try:
            while self.running:
                conn, addr = self.server_socket.accept()
                print("=====================================")
                print(f"   Connected by {addr[0]}:{addr[1]}")
                print("=====================================")
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
        finally:
            self.server_socket.close()

    def _handle_connection(self, conn):
        try:
            while self.running:
                data = self._receive_full_message(conn)
                if not data:
                    break
                self._process_twig(data)
                if self.firsttime:
                    print("=====================================")
                    print(" Received and processed first Twig")
                    print("=====================================")
                    self.firsttime = False
        finally:
            conn.close()

    def _receive_full_message(self, conn):
        # Read the length of the message (first 4 bytes)
        length_data = conn.recv(4)
        print(f"bytes representing the length of the message: {length_data}")
        if not length_data:
            return None
        message_length = int.from_bytes(length_data, byteorder='big') # RFC1700 says network byte order is big-endian
        print(f"message_length: {message_length}")
        # Read the message data based on the length
        data = b''
        while len(data) < message_length:
            print(f"message_length: {message_length}, len(data): {len(data)}, diff: {message_length - len(data)}")
            packet = conn.recv(message_length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _process_twig(self, data):
        twig = Twig().load_bytes(data)
        measurements = self._twig_to_measurements(twig)
        self.global_daemon.add_measurements(twig.coord_sys_id, measurements)

    def _twig_to_measurements(self, twig):
        measurements = []
        for ms in twig.measurement_sets:
            for i, (mean, cov) in enumerate(zip(ms.means, ms.covariances)):
                measurements.append({
                    "type": "NormalMeasurement",
                    "feature": f"{twig.stream_id}_{i}",
                    "mean": mean.tolist(),
                    "covariance": cov.tolist(),
                    "dimensions": ms.dimensions,
                    "is_homogeneous": ms.is_homogeneous
                })
        return measurements

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
