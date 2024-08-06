import xml.etree.ElementTree as ET

def to_feature(feature_id):
    if isinstance(feature_id, FeatureKey):
        return feature_id
    if not isinstance(feature_id, str):
        raise ValueError(f"FeatureKey ID to make a simple StrFeatureKey with feature() must be a string. Got {feature_id}")
    return StrFeatureKey(feature_id)


class FeatureKey:
    def __init__(self):
        self.feature_id = None

    def __hash__(self):
        raise NotImplementedError("FeatureKey objects must implement __hash__")

    def __repr__(self):
        return f"FeatureKey({self.feature_id})"


class StrFeatureKey(FeatureKey):
    def __init__(self, feature_id):
        self.feature_id = feature_id

    def __eq__(self, other):
        return isinstance(other, StrFeatureKey) and self.feature_id == other.feature_id

    def __hash__(self):
        return hash(self.feature_id)

    def __repr__(self):
        return f"StrFeatureKey({self.feature_id})"
    
    def __str__(self):
        return self.feature_id


class FeatureDefinition:
    def __init__(self, feature_xml_str):
        self.tree = ET.ElementTree(ET.fromstring(feature_xml_str))
        self.root = self.tree.getroot()

    def get_measurement_info(self, feature_path):
        element = self._find_element(feature_path)
        if element is None:
            return None

        allowed_measurements = [m.text for m in element.findall('./allowed-measurements/measurement-type')]
        default_measurement = element.find('./default-measurement').text

        return {
            'allowed_measurements': allowed_measurements,
            'default_measurement': default_measurement
        }

    def _find_element(self, path):
        current = self.root
        parts = FeatureDefinitionURI(path).get_xml_path_parts()
        for i, part in enumerate(parts):
            found = False
            for child in current:
                if ((child.tag == 'feature-type' and child.get('name').lower() == part)
                            or child.tag != 'feature-type' and child.tag == part):
                    current = child
                    found = True
                    break
            if not found:
                raise ValueError(f'Error finding element for path "{path}". Element "{part}" not found in feature definition after "/{"/".join(parts[:i])}/"')
        return current


class FeatureDefinitionURI:
    def __init__(self, uri):
        if uri != uri.lower():
            raise ValueError("Feature definition URIs must be lowercase.")
        self.uri = uri
        self._parse_uri()

    def _parse_uri(self):
        # Split the URI into scheme and path
        parts = self.uri.split(':', 1)
        if len(parts) != 2 or parts[0] != 'nestbox':
            raise ValueError("Invalid scheme. Must be 'nestbox'.")
        # Parse the path
        path_parts = parts[1].split('/')
        print(path_parts)
        if len(path_parts) < 5 or path_parts[0] != 'feature' or path_parts[1] != 'def':
            raise ValueError("Invalid feature definition URI format.")
        self.feature_def_id = path_parts[2]
        self.feature_path = '/'.join(path_parts[3:])

    def get_xml_path_parts(self):
        return self.feature_path.split('/')

    def __str__(self):
        return self.uri


if __name__ == "__main__":
    # Example usage:
    try:
        uri = "nestbox:feature/def/fd-hand--v6jg6g60ks9iz23eiz9ykkccsvmvr5qi/hand/root-pose/position"
        fd_uri = FeatureDefinitionURI(uri)
        print("Feature Definition ID:", fd_uri.feature_def_id)
        print("XML Path Parts:", fd_uri.get_xml_path_parts())
    except ValueError as e:
        print("Error:", str(e))

    feature_def = FeatureDefinition("""<feature-definition xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                        xsi:noNamespaceSchemaLocation="feature_schema.xsd">
            <feature-type name="Hand" multiple="true">
                <description>Hand tracking feature</description>
                <root-pose>
                <position dimensions="XYZ">
                    <allowed-measurements>
                    <measurement-type>NormalMeasurement</measurement-type>
                    <measurement-type>OptionsMeasurement</measurement-type>
                    </allowed-measurements>
                    <default-measurement>NormalMeasurement</default-measurement>
                </position>
                <orientation dimensions="IJK">
                    <allowed-measurements>
                    <measurement-type>NormalMeasurement</measurement-type>
                    </allowed-measurements>
                    <default-measurement>NormalMeasurement</default-measurement>
                </orientation>
                </root-pose>
                <features>
                <feature-type name="TrackingPoint" multiple="true">
                    <position dimensions="XYZ">
                    <allowed-measurements>
                        <measurement-type>NormalMeasurement</measurement-type>
                        <measurement-type>CollectionMeasurement</measurement-type>
                    </allowed-measurements>
                    <default-measurement>NormalMeasurement</default-measurement>
                    </position>
                </feature-type>
                </features>
            </feature-type>
            </feature-definition>""")
    hand_position_info = feature_def.get_measurement_info(uri)
    print(hand_position_info)
    # Output: {'allowed_measurements': ['NormalMeasurement', 'OptionsMeasurement'], 'default_measurement': 'NormalMeasurement'}

    finger_position_info = feature_def.get_measurement_info('nestbox:feature/def/fd-hand--v6jg6g60ks9iz23eiz9ykkccsvmvr5qi/hand/features/trackingpoint/position')
    print(finger_position_info)
    # Output: {'allowed_measurements': ['NormalMeasurement', 'CollectionMeasurement'], 'default_measurement': 'NormalMeasurement'}

    # all feature types with multiple="true" must have an identifier or range specifier after the feature name
    # valid identifiers: strings, integer indices, e.g. hand, 0, 1, 2, etc.
    # valid range specifiers: ranges of integers, e.g. 0-23, 0-3, and * for all
    "nestbox:feature/instance/fd-hand--v6jg6g60ks9iz23eiz9ykkccsvmvr5qi/hand/left/root-pose/position" # root pose position for the hand
    "nestbox:feature/instance/fd-hand--v6jg6g60ks9iz23eiz9ykkccsvmvr5qi/hand/left/features/trackingpoint/0-22/position" # exactly 23 tracking points for the left hand
    "nestbox:feature/instance/fd-hand--v6jg6g60ks9iz23eiz9ykkccsvmvr5qi/hand/*/features/trackingpoint/*/position" # all tracking points for all hands
    # example clean URIs for local use
    # feature definition access
    "nestbox:feature/def/hand/root-pose/position"
    # feature instance
    "nestbox:feature/hand/left/root-pose/position" # not ambiguous because feature definitions must have a double dash in their ID, unlike the keywords "instance" and "def"

    # full measurement PUT request
    nestbox.add_measurement("nestbox:feature/hand/left/root-pose/position", type=mtype.NormalMeasurement, mean=[0.1, 0.2, 0.3], covariance=[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
