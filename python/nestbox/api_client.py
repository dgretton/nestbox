from .interfaces import ConnectionConfigInterface
from .networking import ConnectionManager
import json

class NestboxAPIClient:
    def __init__(self, connection_config: ConnectionConfigInterface):
        self.connection_config = connection_config

    def _get_connection(self):
        # for now, use a new connection each time. connection pool or http keep-alive would help efficiency
        conn = ConnectionManager.create_connection(self.connection_config.type, self.connection_config)
        conn.connect()
        return conn

    def create_coordinate_system(self, name=None):
        guid = self._create_coordinate_system()
        if name:
            self.name_coordinate_system(guid, name)
        return guid
    
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
        
    def add_measurements(self, cs_guid, measurements):
        endpoint = f"/coordsys/{cs_guid}/measurements"
        body = json.dumps({"measurements": measurements})
        http_request = self._http_request(endpoint, "POST", body)

        conn = self._get_connection()
        conn.send(http_request.encode('utf-8'))
        
        response = conn.receive().decode('utf-8')
        if not response.startswith('HTTP/1.1 20'):
            raise RuntimeError(f"Failed to add measurements: {response}")
        
        headers, body = response.split('\r\n\r\n', 1)
        return json.loads(body)

    def add_twig(self, cs_guid, twig):
        pass

    def start_alignment(self, cs_guids):
        return self.daemon.handle_alignment_request(cs_guids)

    def close(self):
        self._connection.disconnect()

    # Other methods as defined in your original APIClient class

# {
#     "request_id":"(uuid4)",
#     "type": "set_cs_name",
#     "guid": "unique-guid-string"
#     "name": "new-name"
# }