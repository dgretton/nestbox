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
