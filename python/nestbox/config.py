from .networking import ConnectionConfig
from os import path
import yaml
# go get connection details from ../../config/example_config.yaml
# however, relative import won't work in general, so we need to use the __file__ attribute of the module to get the path to the config file
#TODO: actually, let's not have this be a magic string literal. go get correct connection details from an eventual installed ~/.nestbox/config/... file
with open(path.dirname(__file__) + '/../../config/example_config.yaml', 'r') as stream:
    dconfig = yaml.safe_load(stream)['daemon']
daemon_conn_type = dconfig['server_type']
address = path.expanduser(dconfig['server_address'])
port = dconfig.get('port', None)
DAEMON_CONN_CONFIG = ConnectionConfig(daemon_conn_type, address=address, port=port)
#TODO: This source file config.py will contain some utility functions for getting configs from the correct place modulo detected operating system
