import pyquaternion
import numpy as np
from nestbox.numutil import transform_points, inverse_transform_points, coerce_numpy, coerce_quaternion

class RigidObject: # a rigid object in the environment coordinate system. This is a "cheater" sim-only object that knows its own position and orientation; no coordinate systems should know these details.
    instance_counter = 0

    def __init__(self, origin, orientation=None, feature_points=None, name=None):
        self.origin = origin
        self.orientation = orientation if orientation is not None else pyquaternion.Quaternion(1, 0, 0, 0)
        self.feature_points = feature_points if feature_points is not None else []
        if not name:
            self.name = f'RigidObject{RigidObject.instance_counter}'
            RigidObject.instance_counter += 1
        else:
            self.name = name

    def add_points(self, points):
        self.feature_points.extend(points)

    def get_points(self):
        return transform_points(self.origin, self.orientation, self.feature_points)
    
    def get_feature_ids(self):
        return [f'{self.name}_pt{i}' for i in range(len(self.feature_points))]

    def get_points_dict(self):
        return {fid: pt for fid, pt in zip(self.get_feature_ids(), self.get_points())}


class GroundTruthCoordinateSystem:
    def __init__(self, coord_sys, origin, orientation):
        self.coord_sys = coord_sys
        self.origin = coerce_numpy(origin)
        self.orientation = coerce_quaternion(orientation)

class SimEnvironment:
    def __init__(self):
        self.rigidobjects = []
        self._ground_truths_map = {} # a dictionary of observers and their ground truths. the key is the observer

    def add_rigidobject(self, obj):
        self.rigidobjects.append(obj)

    def place_coordinate_system(self, coord_sys, actual_origin, actual_orientation=pyquaternion.Quaternion(1, 0, 0, 0)):
        gt = GroundTruthCoordinateSystem(coord_sys, actual_origin, actual_orientation)
        for observer in coord_sys.observers:
            self._ground_truths_map[observer] = gt

    def points_from_observer_perspective(self, observer, points):
        gt = self._ground_truths_map[observer]
        points = inverse_transform_points(gt.origin, gt.orientation, points)
        points = inverse_transform_points(observer.position, observer.orientation, points)
        return points

    def points_dict_from_observer_perspective(self, observer, points_dict):
        if not points_dict:
            return {}
        feature_ids, points = zip(*points_dict.items())
        return {feature_id: pt for feature_id, pt in zip(feature_ids, self.points_from_observer_perspective(observer, points))}
    
    def ground_truths(self):
        return list(set(self._ground_truths_map.values()))

    def project_to_image(self, camera, points_dict):
        # Assume the points are not already transformed into the camera's coordinate system.
        obs_points_dict = self.points_dict_from_observer_perspective(camera, points_dict)

        # Filter out points where z <= 0 (behind camera)
        # valid_points = points[points[:, 2] > 0]
        valid_points_dict = {feature_id: pt for feature_id, pt in obs_points_dict.items() if pt[2] > 0}
        valid_features, valid_points = zip(*valid_points_dict.items())
        valid_points = coerce_numpy(valid_points)

        # Calculate the angles in radians within the camera's field of view
        angles = np.arctan2(valid_points[:, 0], valid_points[:, 2]), np.arctan2(valid_points[:, 1], valid_points[:, 2])

        # Normalize angles to [-1, 1] based on the sensor size
        img_space_x = angles[0] / (camera.sensor_size[0] / 2)
        img_space_y = angles[1] / (camera.sensor_size[1] / 2)

        # Create an array of normalized image space coordinates
        img_space_angles = np.vstack((img_space_x, img_space_y)).T

        # Filter out points outside the sensor's field of view
        in_view = np.logical_and(np.abs(img_space_angles[:, 0]) <= 1, np.abs(img_space_angles[:, 1]) <= 1)
        img_space_angles = img_space_angles[in_view]

        assert len(img_space_angles) == len(valid_points) # just for now to be sure we're not losing any points while we're still matching points to measurements by index

        return {feature_id: img_space_angle for feature_id, img_space_angle in zip(valid_features, img_space_angles)}


