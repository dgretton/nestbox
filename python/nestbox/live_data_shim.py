import socket
import threading
from nestbox.interfaces import ConnectionInterface
from nestbox.protos import Twig, Dim
from nestbox.measurement import BearingMeasurement
from nestbox.feature import to_feature
from nestbox.handedness import mirror_point, mirror_covariance

class LiveDataShim:
    """Receives length-prefixed Twigs from the Quest (NestboxHand) and feeds them to the daemon.

    Each hand Twig carries the raw OVRSkeleton.Bones positions in Unity world space, in OVR bone-ID order
    (0 WristRoot ... 20 IndexTip, 21-23 Middle/Ring/PinkyTip). Points are named by bone ID, e.g.
    nestbox:feature/hand/righthand/bone/20/position, and mirrored from Unity's left-handed frame (x -> -x).
    """
    def __init__(self, connection: ConnectionInterface, mirror_x=True):
        from nestbox.daemon import global_daemon
        self.global_daemon = global_daemon
        self.host = connection.connection_info['host'] # for simplicity for this patch just reconstruct the connection from the connection info
        self.port = connection.connection_info['port']
        self.server_socket = None
        self.running = False
        self.firsttime = True
        self.mirror_x = mirror_x

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
        length_data = b''
        while len(length_data) < 4:
            packet = conn.recv(4 - len(length_data))
            if not packet:
                return None
            length_data += packet
        message_length = int.from_bytes(length_data, byteorder='big') # RFC1700 says network byte order is big-endian
        # Read the message data based on the length
        data = b''
        while len(data) < message_length:
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
        for ms in twig.measurement_sets: # better hope there is only one measurement set
            for bone_id, (mean, cov) in enumerate(zip(ms.means, ms.covariances)):
                if self.mirror_x:
                    mean, cov = mirror_point(mean), mirror_covariance(cov)
                feature = f"nestbox:feature/hand/{twig.stream_id}/bone/{bone_id}/position"
                if ms.dimensions == [Dim.I, Dim.J, Dim.K]:
                    measurements.append(BearingMeasurement.from_unit_vector(
                        to_feature(feature), mean, cov, dimensions=ms.dimensions,
                        is_homogeneous=ms.is_homogeneous).to_dict())
                    continue
                measurements.append({
                    "type": "NormalMeasurement",
                    "feature": f"nestbox:feature/hand/{twig.stream_id}/bone/{bone_id}/position",
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
