import numpy as np
import json
from cube import vertices as cube_vertices, edges as cube_edges
from coordsystem import transform_points, quaternion_to_basis, transform_point, rotate_covariance, CameraObserver, PointTrackerObserver, coerce_quaternion, coerce_numpy
from pyquaternion import Quaternion

def qprod(q1, q2):
    return coerce_quaternion(q1) * coerce_quaternion(q2)

def transform_points_zipped(origin, quaternion, x, y, z):
    return list(zip(*transform_points(origin, quaternion, list(zip(x, y, z)))))

def quaternion_dict(quaternion): # because other systems e.g. threejs use a different order convention
    return {k:float(v) for k, v in zip('wxyz', quaternion)}

def make_serializable(data):
    for key, value in data.items():
        if isinstance(value, Quaternion):
            data[key] = quaternion_dict(value)
            continue
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        if hasattr(value, '__iter__'):
            try:
                data[key] = [float(x) for x in data[key]]
            except TypeError:
                continue
            except ValueError:
                continue
    return data

class VisualElement:
    def __init__(self, element_type, **kwargs):
        self.name = None
        self.type = element_type
        # ensure serialization
        make_serializable(kwargs)
        self.properties = kwargs
        self.parent_name = None

    def set_parent(self, parent):
        self.parent_name = parent

    def update(self, **kwargs):
        self.properties.update(kwargs)

    def state(self):
        return {'name': self.name, 'type': self.type, 'parent':self.parent_name, 'properties': make_serializable(self.properties)}

class Visualizer:
    def __init__(self, aligner, environment):
        self.aligner = aligner
        self.environment = environment
        self._vis_elements = {}

    def cached_vis_element(self, name):
        def wrap(f):
            def wrapper(*args, **kwargs):
                if name in self._vis_elements:
                    return self._vis_elements[name]
                vis_element = f(*args, **kwargs)
                self._vis_elements[name] = vis_element
                vis_element.name = name
                return vis_element
            return wrapper
        return wrap

    def plot_cube(self, name, position=(0, 0, 0), orientation=Quaternion(1, 0, 0, 0), size=1, parent_coord_sys=None, line_width=1, color=(1, 1, 1), opacity=1):
        # wireframe cube
        @self.cached_vis_element(name)
        def cube_element(position, orientation, size, line_width, color, opacity):
            ve = VisualElement('cube', position=position, orientation=orientation, size=size, line_width=line_width, color=color, opacity=opacity)
            if parent_coord_sys is not None:
                ve.set_parent(parent_coord_sys)
            return ve
        cube_element(position, orientation, size, line_width, color, opacity).update(position=position, orientation=orientation)

    def plot_covariance_ellipsoid(self, name, mean, covariance, parent_coord_sys=None, color=(0.5, 0.5, 0.9), opacity=0.2):
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            print(f"eigenvectors failed for {name} with covariance {covariance}")
            return

        # Plotting the transformed ellipsoid
        @self.cached_vis_element(name)
        def ellipsoid_element(position, eigenvalues, eigenvectors, color, opacity):
            ve = VisualElement('ellipsoid', position=position, eigenvalues=eigenvalues, eigenvectors=eigenvectors, color=color, opacity=opacity)
            if parent_coord_sys is not None:
                ve.set_parent(parent_coord_sys)
            return ve
        ellipsoid_element(mean, eigenvalues, eigenvectors, color, opacity).update(position=mean, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def plot_camera(self, name, position, orientation, parent_coord_sys=None, size=.5): # TODO: currently does not show the actual field of view
        """
        Plots a simple representation of a camera in 3D space.
        """
        # Camera's pointing direction
        end_point = (0, 0, 1)
        root_point = (0, 0, 0)

        def plot_camera_lines(name, x, y, z, color=(1, 0, 0), line_width=2):
            @self.cached_vis_element(name)
            def camera_lines_element(x, y, z, color, line_width):
                line = VisualElement('line', position=position, orientation=orientation, x=x, y=y, z=z, color=color, line_width=line_width)
                if parent_coord_sys is not None:
                    line.set_parent(parent_coord_sys)
                return line
            trans_x, trans_y, trans_z = transform_points_zipped(position, orientation, x, y, z)
            camera_lines_element(trans_x, trans_y, trans_z, color, line_width).update(x=trans_x, y=trans_y, z=trans_z)

        # Draw the direction line
        plot_camera_lines(f"camera {name} direction line", [root_point[0], end_point[0]], [root_point[1], end_point[1]], [root_point[2], end_point[2]])

        # Create a pyramid to represent the camera's field of view
        base_center = (0, 0, size)
        angles = np.linspace(np.pi/4, (2 + 1/4) * np.pi, 5)
        x_base = base_center[0] + size * np.cos(angles)
        y_base = base_center[1] + size * np.sin(angles)
        z_base = np.array([base_center[2]] * 5)

        # Connect base points to the camera position
        for i in range(4):
            plot_camera_lines(f"camera {name} pyramid line {i}", [x_base[i], 0], [y_base[i], 0], [z_base[i], 0])

        # Close the base
        plot_camera_lines(f"camera {name} base lines", x_base, y_base, z_base)

    def plot_point_collection(self, name, points, parent_coord_sys=None, color=(1, 1, 1), marker_size=0.05):
        points = coerce_numpy(points)
        x, y, z = points.T
        @self.cached_vis_element(name)
        def point_collection_element(**properties):
            ve = VisualElement('point_collection', **properties) # TODO: This is done in this slightly roundabout way in anticipation of potentially expensive setup for visual elements like reading geometry from disk. For now it doesn't help much.
            if parent_coord_sys is not None:
                ve.set_parent(parent_coord_sys)
            return ve
        point_collection_element(x=x, y=y, z=z, color=color, marker_size=marker_size).update(x=x, y=y, z=z)

    def draw(self):
        # Plot the rigid objects at their actual positions in the environment
        for rigidobject in self.environment.rigidobjects:
            self.plot_point_collection(f"rigidobject {rigidobject}", rigidobject.get_points())
            self.plot_cube(f"rigidobject {rigidobject} box", rigidobject.origin, rigidobject.orientation)

        # Plot the observers and show measurements as ellipsoids.
        for coord_sys, origin, orientation in self.aligner.iterate_coordinate_systems():
            for i, (mean, covariance) in enumerate(coord_sys.measurements):
                self.plot_covariance_ellipsoid(f"coordinate system {coord_sys} measurement {i} ellipsoid", mean, covariance, parent_coord_sys=coord_sys.name, color=(0, 1, 1), opacity=.2)
            #meas_means = [meas[0] for meas in coord_sys.measurements]
            #self.plot_point_collection(f"coordinate system {coord_sys.name} measurements", meas_means, parent_coord_sys=coord_sys.name, color=(0, 1, 1), marker_size=0.05)
            for observer in coord_sys.observers:
                if isinstance(observer, PointTrackerObserver):
                     self.plot_cube(f"tracker {observer} box", observer.position, observer.orientation, parent_coord_sys=coord_sys.name, color=(0, 1, 1), line_width=.5)
                if isinstance(observer, CameraObserver):
                    #observer has observer.position, observer.orientation
                    #we need to place it relative to the coordinate system's origin and orientation, meaning that its position and orientation need to be modified before plotting
                    self.plot_camera(f"camera {observer}", observer.position, observer.orientation, parent_coord_sys=coord_sys.name)
                    self.plot_point_collection(f"camera corners {observer}", observer.image_corners(), parent_coord_sys=coord_sys.name, color=(1, 0, 0), marker_size=0.05)


    def state(self):
        state = {}
        coordinate_systems = []
        for i, (coord_sys, origin, orientation) in enumerate(self.aligner.iterate_coordinate_systems()):
            coordinate_system_data = {}
            coordinate_system_data['name'] = coord_sys.name
            coordinate_system_data['basis'] = make_serializable({
                'origin': origin,
                'orientation': quaternion_dict(orientation)
            })
            observers = []
            for obs in coord_sys.observers:
                observer = make_serializable({'position': obs.position, 'orientation': quaternion_dict(obs.orientation)})
                if isinstance(obs, CameraObserver):
                    observer.update({
                        'type': 'camera',
                        'properties': make_serializable({
                            'focalDistance': obs.focal_distance,
                            'depthOfField': obs.depth_of_field,
                            'sensorSize': obs.sensor_size,
                            'resolution': obs.resolution
                        }),
                    })
                elif isinstance(obs, PointTrackerObserver):
                    observer.update({
                        'type': 'tracker',
                        'properties': {
                            'variance': obs.variance
                        }
                    })
                observers.append(observer)
            coordinate_system_data['observers'] = observers

            coordinate_systems.append(coordinate_system_data)
        state['coordinateSystems'] = coordinate_systems
        state['visualElements'] = [ve.state() for ve in self._vis_elements.values()]
        return state
