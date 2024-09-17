from .interfaces import ConnectionConfigInterface
from .networking import ConnectionManager
from .aligner import AlignmentResult
import requests
from requests.adapters import BaseAdapter
from requests.models import Response
from urllib.parse import unquote, quote
from typing import Any, Dict
import json
import time

class NestboxAPIClient:
    def __init__(self, connection_config: ConnectionConfigInterface):
        self.session = requests.Session()
        adapter = CustomConnectionAdapter(ConnectionManager, connection_config)
        self.session.mount('http+unix://', adapter)

        # URL-encode the Unix socket path
        socket_path = quote(connection_config.address, safe='')
        self.base_url = f'http+unix://{socket_path}'

    def create_coordinate_system(self, name=None):
        url = f'{self.base_url}/coordsys'
        response = self.session.post(url, json={"type": "create_cs"})
        response.raise_for_status()
        guid = response.json()['cs_guid']
        #print(f"API client received GUID from create_coordinate_system: {guid}")
        if name:
            self.name_coordinate_system(guid, name)
        return guid

    def name_coordinate_system(self, cs_guid, name):
        url = f'{self.base_url}/coordsys/{cs_guid}/name'
        response = self.session.put(url, json={"name": name})
        #response.raise_for_status()
        #print(f"API client received response from name_coordinate_system: {response}")
        return response.json()

    def add_normal_measurement(self, feature, cs, mean, covariance, dimensions, is_homogeneous):
        url = f'{self.base_url}/coordsys/{cs}/measurement'
        response = self.session.post(url, json={
            "type": "NormalMeasurement",
            "feature": feature,
            "mean": mean,
            "covariance": covariance,
            "dimensions": dimensions,
            "is_homogeneous": is_homogeneous
        })
        response.raise_for_status()
        #print(f"API client received response from add_normal_measurement: {response}")
        return response.json()

    def add_measurements(self, cs, measurements):
        url = f'{self.base_url}/coordsys/{cs}/measurements'
        response = self.session.post(url, json={"measurements": measurements})
        response.raise_for_status()
        #print(f"API client received response from add_measurements: {response}")
        return response.json()

    def start_alignment(self):
        url = f'{self.base_url}/alignment'
        response = self.session.post(url, json={"action": "start"})
        response.raise_for_status()
        #print(f"API client received response from start_alignment: {response}")
        return response.json()

    def get_transform(self, source_cs: str, target_cs: str, relation_type: str='convert') -> AlignmentResult:
        url = f'{self.base_url}/transforms/{relation_type}/{source_cs}/to/{target_cs}'
        response = self.session.get(url)
        response.raise_for_status()
        #print(f"API client received response from get_transform: {response}")
        return AlignmentResult.from_json(response.json())

    def get_current_measurement(self, cs_guid: str, feature_id: str) -> Dict[str, Any]:
        url = f'{self.base_url}/coordsys/{cs_guid}/measurement?feature={quote(feature_id)}'
        response = self.session.get(url)
        response.raise_for_status()
        #print(f"API client received response from get_current_measurement: {response}")
        return response.json()

    def create_stream(self, config):
        url = f'{self.base_url}/stream'
        response = self.session.post(url, json=config)
        response.raise_for_status()
        #print(f"API client received response from create_stream: {response}")
        return response.json()

    def send_twig(self, twig_data):
        url = f'{self.base_url}/twig'
        response = self.session.post(url, json=twig_data)
        response.raise_for_status()
        #print(f"API client received response from send_twig: {response}")
        return response.json()

    def set_router(self, stream_id, config):
        url = f'{self.base_url}/stream/{stream_id}/router'
        response = self.session.post(url, json=config)
        response.raise_for_status()
        #print(f"API client received response from set_router: {response}")
        return response.json()

    def close(self):
        self._conn.disconnect()


class CustomConnectionAdapter(BaseAdapter):
    def __init__(self, connection_manager: ConnectionManager, connection_config: ConnectionConfigInterface):
        self.connection_manager = connection_manager
        self.connection_config = connection_config

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        start_time = time.time()
        # Parse the URL
        authority, path = request.url.split('://')
        quoted_socket_path, *path_parts = path.split('/')
        socket_path = unquote(quoted_socket_path) # decode the URL-encoded socket path
        endpoint = '/' + '/'.join(path_parts)
        #print(f"CustomConnectionAdapter.send: socket_path is {socket_path}, endpoint is {endpoint}")

        # Use the socket_path to create your connection
        conn = self.connection_manager.create_connection(self.connection_config.type, self.connection_config)
        try:
            conn.connect()
        except FileNotFoundError: #TODO: catch other types of exception or raise custom exception inside connect()
            raise ConnectionError("Is the nestbox daemon running?")

        # Convert requests' request to your format
        body = request.body.decode('utf-8') if request.body else ""
        headers = request.headers
        #print(f"CustomConnectionAdapter.send: headers are {headers}, body is {body}")

        # Create your custom HTTP request
        http_request = f"{request.method} {endpoint} HTTP/1.1\r\n"
        for key, value in headers.items():
            http_request += f"{key}: {value}\r\n"

        http_request += f"\r\n{body}"

        # Send and receive using your connection
        conn.send(http_request.encode('utf-8'))
        raw_response = conn.receive().decode('utf-8')

        # Parse the raw response
        status_line, rest = raw_response.split('\r\n', 1)
        headers, body = rest.split('\r\n\r\n', 1)
        #print(f"CustomConnectionAdapter.send response: received response: status_line is {status_line}, headers are:\n\n{headers}\n")

        # parse out content-length header
        content_length = int(headers.split('Content-Length: ')[1].split('\r\n')[0])
        while len(body) < content_length and (timeout is None or time.time() - start_time < timeout):
            body += conn.receive().decode('utf-8')
        #print(f"CustomConnectionAdapter.send response: body is {body}")

        # Create a requests Response object
        response = Response()
        response.status_code = int(status_line.split()[1])
        response.headers = dict(line.split(': ') for line in headers.split('\r\n'))
        response._content = body.encode('utf-8')

        return response

    def close(self):
        pass
