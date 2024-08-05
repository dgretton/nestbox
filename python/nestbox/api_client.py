from .interfaces import ConnectionConfigInterface
from .networking import ConnectionManager
from .aligner import AlignmentResult
import json

class NestboxAPIClient:
    def __init__(self, connection_config: ConnectionConfigInterface):
        self.connection_config = connection_config
        self._conn = None

    def _get_connection(self):
        # for now, use a new connection each time. connection pool or http keep-alive would help efficiency
        if self._conn and self._conn.is_connected():
            self._conn.disconnect()
        self._conn = ConnectionManager.create_connection(self.connection_config.type, self.connection_config)
        self._conn.connect()
        return self._conn

    def _http_request(self, endpoint: str, method: str, body: str) -> str:
        assert method in ['GET', 'POST', 'PUT', 'DELETE']
        headers = {
            "Content-Type": "application/json",
            "Connection": "close"  # Important for some servers
        }
        http_request = (
            f"{method} {endpoint} HTTP/1.1\r\n"
            f"Host: localhost\r\n"
            f"Content-Length: {len(body)}\r\n"
        )
        for header, value in headers.items():
            http_request += f"{header}: {value}\r\n"
        http_request += f"\r\n{body}"
        return http_request

    def create_coordinate_system(self, name=None):
        while True: # TODO This is most certainly NOT the right behavior long-term!
                    # TODO This is just a temporary workaround because create cs keeps returning an empty body randomly
                    # TODO fix that, then this.
            try:
                guid = self._create_coordinate_system()
                break
            except Exception as e:
                print(f"Error creating coordinate system, retrying: {e}")
        if name:
            self.name_coordinate_system(guid, name)
        return guid

    def _create_coordinate_system(self):
        endpoint = "/coordsys"
        body = json.dumps({"type": "create_cs"})
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))
        
        # Now we need to parse the HTTP response
        response = conn.receive().decode('utf-8')
        print(f"API client received response from create_coordinate_system: {response}")
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to create coordinate system: {response}")
        
        # Split the response into headers and body
        headers, body = response.split('\r\n\r\n', 1)
        print(f"API client received body from create_coordinate_system: {body}")
        
        if not body.strip():
            raise RuntimeError("Empty body received from create_coordinate_system")
        # Parse the JSON body
        json_response = json.loads(body)
        
        return json_response['cs_guid']

    def name_coordinate_system(self, cs_guid, name):
        # "type": "name_cs",
        # "cs_guid": "unique-guid-string"
        # "name": "new-name"
        endpoint = f"/coordsys/{cs_guid}/name"
        body = json.dumps({"type": "name_cs", "name": name})
        http_request = self._http_request(endpoint, "PUT", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))

        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to name coordinate system: {response}")
    
    def add_normal_measurement(self, feature, cs, mean, covariance, dimensions, is_homogeneous):
        endpoint = f"/coordsys/{cs}/measurement"
        body = json.dumps({
            "type": "NormalMeasurement",
            "feature": feature,
            "mean": mean,
            "covariance": covariance,
            "dimensions": dimensions,
            "is_homogeneous": is_homogeneous})
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))
        
        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to add measurement: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json 

    def add_measurements(self, cs, measurements):
        endpoint = f"/coordsys/{cs}/measurements"
        body = json.dumps({"measurements": measurements})
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))
        
        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to add measurements: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def start_alignment(self):
        endpoint = "/alignment"
        body = json.dumps({"action": "start"})
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))
        
        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to start alignment: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def get_transform(self, source_cs: str, target_cs: str, relation_type: str='convert') -> AlignmentResult:
        endpoint = f"/transforms/{relation_type}/{source_cs}/to/{target_cs}"
        http_request = self._http_request(endpoint, "GET", "")

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))

        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to get transform: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        print(body)
        return AlignmentResult.from_json(json.loads(body))

    # Stream methods
    def create_stream(self, config):
        endpoint = "/stream"
        body = json.dumps(config)
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))

        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to create stream: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def send_twig(self, twig_data):
        endpoint = f"/twig"
        body = json.dumps(twig_data)
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))

        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to send twig: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def set_router(self, stream_id, config):
        endpoint = f"/stream/{stream_id}/router"
        body = json.dumps(config)
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))

        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to create router: {response}")

        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def close(self):
        self._connection.disconnect()

    # Other methods as defined in your original APIClient class

# {
#     "request_id":"(uuid4)",
#     "type": "set_cs_name",
#     "guid": "unique-guid-string"
#     "name": "new-name"
# }