# NestboxPublicAPI (ClientAPI) implementation
class NestboxAPIClient:
    def __init__(self):
        pass

    def create_coordinate_system(self, name):
        pass

    def name_coordinate_system(self, cs_id, name):
        pass

    def add_measurement_set(self, cs_id, twig_data):
        pass

    def start_alignment(self, cs_guids):
        return self.daemon.handle_alignment_request(cs_guids)

    # Other methods as defined in your original APIClient class

# {
#     "request_id":"(uuid4)",
#     "type": "set_cs_name",
#     "guid": "unique-guid-string"
#     "name": "new-name"
# }