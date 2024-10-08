import pyquaternion
import numpy as np
from nestbox.numutil import transform_point, transform_points, rotate_covariance, coerce_numpy
from nestbox.feature import to_feature
from nestbox.measurement import Measurement, NormalMeasurement

class CoordinateSystem:
    names = set()

    def __init__(self, name=None):
        self.observers = []
        self.measurements = {} # map of features to their measurements
        self.stale = True
        if name is None:
            name = "CoordinateSystem0"
            i = 1
            while name in CoordinateSystem.names:
                name = f"CoordinateSystem{i}"
                i += 1
        elif name in CoordinateSystem.names:
            raise ValueError(f"Coordinate system with name {name} already exists")
        CoordinateSystem.names.add(name)
        self.name = name

    def add_local_observer(self, observer):
        self.observers.append(observer)

    def measurements_map(self):
        return self.measurements

    def update_measurements(self, measurements):
        print(f"Coordinate system {self.name} updating measurements")
        if not measurements:
            print("No measurements given.")
        for m in measurements:
            print(f"Adding measurement {m} for feature {m.feature}")
        if not all(isinstance(m, Measurement) for m in measurements):
            raise ValueError("All measurements must be Measurement objects")
        self.set_stale() # mark that the model will now need to be rebuilt before more optimization can happen
        self.measurements.update({m.feature: m for m in measurements})

    def get_measurement(self, feature):
        return self.measurements.get(feature)

    def clear_measurements(self, clear_key=None):
        self.measurements = {feature: measurement for feature, measurement in self.measurements.items() if not measurement.clear_key.startswith(clear_key)}

    def clear_all_measurements(self):
        self.measurements = {}

    def set_stale(self, stale=True):
        self.stale = stale


class Observer:
    '''
    base class for:
        - CameraObserver
        - PointTrackerObserver
        - PoseObserver
    Both will create (mean, covariance)-type measurements.
    Observers can have a position and orientation relative to the coordinate system they represent, e.g. two very rigidly coupled cameras for binocular vision.
    Cameraobserver will give very precise angular measurements and poor depth precision.
    PointTrackerObserver will give reasonably precise 3D coordinate measurements of points but not 3D poses. Since it's a tracking space, it might reorient or translate itself, but over short time intervals it should give good absolute positioning.
    PoseObserver gives a full characterization of the pose of the object, including its position and orientation, but can be expected to have a fair amount of noise in both.
    '''
    def __init__(self, position=(0, 0, 0), orientation=pyquaternion.Quaternion(1, 0, 0, 0)):
        self.position = position
        self.orientation = orientation

    def forward(self):
        return transform_point(self.position, self.orientation, [0, 0, 1])

    def measure(self):
        raise NotImplementedError("Subclasses must implement this method")


class PointTrackerObserver(Observer):
    def __init__(self, position=(0, 0, 0), orientation=pyquaternion.Quaternion(1, 0, 0, 0), variance=1.0):
        super().__init__(position, orientation)
        self.variance = variance

    def measure(self, points_dict):
        return [NormalMeasurement(
            to_feature(feature),
            transform_point(self.position, self.orientation, point),
            coerce_numpy(np.eye(3) * self.variance))
            for feature, point in points_dict.items()]


class CameraObserver(Observer):
    def __init__(self, position=(0, 0, 0), orientation=pyquaternion.Quaternion(1, 0, 0, 0), sensor_size=(np.pi/4, np.pi/4), focal_distance=10, depth_of_field=5, resolution=(1280, 720)):
        # sensor_size is the field of view of the sensor in radians
        # focal_distance is the approximate distance from the camera to the focal plane, which will help determine the means of the depth measurements
        # depth_of_field is the approximate length scale of the camera, which will be used as the standard deviation to determine the covariance of the depth measurement
        # resolution is the number of effective pixels in the image, that is, the number of points lined up across the axis that can be resolved--probably quite a lot less than the actual number of pixels in the image.
        super().__init__(position, orientation)
        self.sensor_size = sensor_size
        self.focal_distance = focal_distance
        self.depth_of_field = depth_of_field
        self.resolution = resolution

    def measure(self, img_space_angles_dict):
        # img_space_angles_dict is a dictionary of feature ids or feature objects to their angles in the image space, measured in the range [-1, 1] in both dimensions.
        print(img_space_angles_dict, "img_space_angles_dict")
        features = []
        img_space_angles = []
        for feature, angle in img_space_angles_dict.items():
            features.append(to_feature(feature))
            img_space_angles.append(angle)

        # scale img_space_angles to actual angles using the camera sensor size.  Also, the zero-point for phi is pi/2, not 0, to avoid pole singularity at phi=0.
        angles = np.array(img_space_angles) * np.array(self.sensor_size) / 2 + np.array([0, np.pi/2])

        # Transform means to the coordinate system space
        means_coord_sys_space = coerce_numpy(self.image_space_angles_to_coord_space(img_space_angles)) # coerce is for data type

        # Define spherical covariance matrix
        cov_spherical = np.diag([
            self.depth_of_field**2,  # Large variance in radial direction
            (self.sensor_size[0] / self.resolution[0])**2,  # Small variance in angular theta direction
            (self.sensor_size[1] / self.resolution[1])**2  # Small variance in angular phi direction
        ])

        # List to collect Cartesian covariances
        covs_coord_sys_space = []

        # Convert each point's covariance from spherical to Cartesian
        for theta, phi in angles:
            # Calculate the Jacobian matrix for the transformation
            J = np.array([
                [np.sin(phi) * np.sin(theta), self.focal_distance * np.sin(phi) * np.cos(theta), self.focal_distance * np.cos(phi) * np.sin(theta)],
                [-np.cos(phi), 0, self.focal_distance * np.sin(phi)],
                [np.sin(phi) * np.cos(theta), -self.focal_distance * np.sin(phi) * np.sin(theta), self.focal_distance * np.cos(phi) * np.cos(theta)]
            ])

            # Compute the Cartesian covariance matrix for the point
            cov_cartesian = J @ cov_spherical @ J.T

            # Transform the covariance matrix to the coordinate system space
            cov_cartesian = rotate_covariance(self.orientation, cov_cartesian)
            covs_coord_sys_space.append(coerce_numpy(cov_cartesian)) # coerce is for data type

        print(means_coord_sys_space, "means_coord_sys_space")
        print(covs_coord_sys_space, "covs_coord_sys_space")
        # add measurement means and covariances in coordinate system space
        return [NormalMeasurement(feature, mean, cov) for feature, mean, cov in zip(features, means_coord_sys_space, covs_coord_sys_space)]

    def image_corners(self):
        img_space_angles = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        return self.image_space_angles_to_coord_space(img_space_angles)

    def image_space_angles_to_camera_space(self, img_space_angles):
        angles = np.array(img_space_angles) * np.array(self.sensor_size) / 2 + np.array([0, np.pi/2])
        return np.array([
            np.tan(angles[:, 0]) * self.focal_distance,
            np.tan(angles[:, 1]-(np.pi/2)) * self.focal_distance,
            self.focal_distance * np.cos(angles[:, 0]) * np.sin(angles[:, 1])
        ]).T

    def image_space_angles_to_coord_space(self, img_space_angles):
        return transform_points(self.position, self.orientation, self.image_space_angles_to_camera_space(img_space_angles))
