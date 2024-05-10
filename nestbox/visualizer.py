from mayavi import mlab as mlab_module
import numpy as np
from cube import vertices as cube_vertices, edges as cube_edges
from coordsystem import transform_points, quaternion_to_basis, transform_point, rotate_covariance, CameraObserver, PointTrackerObserver, coerce_quaternion
from pyquaternion import Quaternion
import time

def qprod(q1, q2):
    return coerce_quaternion(q1) * coerce_quaternion(q2)

#replace mlab with a wrapper that always prints whatever functions are called on it
class MlabWrapper:
    def __getattr__(self, name):
        def f(*args, **kwargs):
            print(f"mlab.{name}({args}, {kwargs})")
            return mlab_module.__getattribute__(name)(*args, **kwargs)
        return f
    
# mayavi start/stop render context manager
class RenderOff:
    def __enter__(self):
        mf = mlab_module.gcf()
        if mf is not None:
            mf.scene.disable_render = True
        return self

    def __exit__(self, type, value, traceback):
        mf = mlab_module.gcf()
        if mf is not None:
            mf.scene.disable_render = False
    
mlab = MlabWrapper()

def transform_points_zipped(origin, quaternion, x, y, z):
#     print("X, Y, Z:", x, y, z)
#     print(list(zip(x, y, z)))
#     print("origin:", origin)
#     print("quaternion:", quaternion)
#     print("Transform points:", transform_points(origin, quaternion, list(zip(x, y, z))))
    return list(zip(*transform_points(origin, quaternion, list(zip(x, y, z)))))

class Visualizer:
    def __init__(self, aligner, environment):
        self.aligner = aligner
        self.environment = environment
        self.closed = False
        self._element_source_data = {}

    def cache_source_under(self, name):
        def wrap(f):
            def wrapper(*args, **kwargs):
                if name in self._element_source_data:
                    return self._element_source_data[name]
                mlab_source = f(*args, **kwargs)
                self._element_source_data[name] = mlab_source
                return mlab_source
            return wrapper
        return wrap

    def plot_bases(self):
        for i, (_, origin, orientation) in enumerate(self.aligner.iterate_coordinate_systems()):

            def quiver(name, origin, direction, color):
                @self.cache_source_under(name)
                def quiver_source_data(origin, direction, color):
                    return mlab.quiver3d(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], color=color, mode='arrow', scale_factor=1).mlab_source
                quiver_source_data(origin, direction, color).reset(x=origin[0], y=origin[1], z=origin[2], u=direction[0], v=direction[1], w=direction[2])

            x_dir, y_dir, z_dir = quaternion_to_basis(orientation)
            quiver(f"basis {i} x", origin, x_dir, (1, 0, 0))
            quiver(f"basis {i} y", origin, y_dir, (0, 1, 0))
            quiver(f"basis {i} z", origin, z_dir, (0, 0, 1))

    def plot_box(self, name, scale=1, position=(0, 0, 0), orientation=Quaternion(1, 0, 0, 0), color=(1, 1, 1)):
        # Plot a cube made of thin colored lines
        # USE SPARINGLY. for some reason this is the biggest slowdown in the whole project

        def plot_line(name, p1, p2):
            p1 = transform_point(position, orientation, p1)
            p2 = transform_point(position, orientation, p2)
            @self.cache_source_under(name)
            def line_source_data(p1, p2, color):
                return mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, tube_radius=None, line_width=1, opacity=0.3).mlab_source
            line_source_data(p1, p2, color).reset(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]])

        edges = [np.array(edge)-.5 for edge in cube_edges]
        for i, edge in enumerate(edges):
            plot_line(f"box {name} edge {i}", *(edge*scale))

    def plot_covariance_ellipsoid(self, name, mean, covariance, color=(0.5, 0.5, 0.9), opacity=0.2):
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            print(f"eigenvectors failed for {name} with covariance {covariance}")
            return

        # Generate a sphere
        phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Scale the sphere to the ellipsoid using eigenvalues and rotate it
        ellipsoid = np.stack([x, y, z], axis=-1) * np.sqrt(eigenvalues)
        ellipsoid = np.dot(ellipsoid, eigenvectors.T)

        # Translate the ellipsoid to the mean
        ellipsoid = ellipsoid + mean[np.newaxis, np.newaxis, :]

        # Plotting the transformed ellipsoid
        @self.cache_source_under(name)
        def ellipsoid_source_data(ellipsoid, color, opacity):
            return mlab.mesh(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2], color=color, opacity=opacity).mlab_source
        ellipsoid_source_data(ellipsoid, color, opacity).set(x=ellipsoid[:, :, 0], y=ellipsoid[:, :, 1], z=ellipsoid[:, :, 2])

    def plot_camera(self, name, position, orientation, size=.5): # TODO: currently does not show the actual field of view
        """
        Plots a simple representation of a camera in 3D space.
        """
        # TODO: make animate-able like other mlab_source stuff
        # Camera's pointing direction
        end_point = (0, 0, 1)
        root_point = (0, 0, 0)

        def plot_camera_lines(name, x, y, z, color=(1, 0, 0), tube_radius=None, line_width=2):
            @self.cache_source_under(name)
            def camera_lines_source_data(x, y, z, color, tube_radius, line_width):
                return mlab.plot3d(x, y, z, color=color, tube_radius=tube_radius, line_width=line_width).mlab_source
            trans_x, trans_y, trans_z = transform_points_zipped(position, orientation, x, y, z)
            camera_lines_source_data(trans_x, trans_y, trans_z, color, tube_radius, line_width).reset(x=trans_x, y=trans_y, z=trans_z)
            #(*transform_points_zipped(position, orientation, x, y, z), color=color, tube_radius=tube_radius, line_width=line_width)

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

    def plot_point_collection(self, name, points, color=(1, 1, 1), scale_factor=0.1):
        # as a note, the shape of the points array may change depending on the number of points, relevant for mlab
        points = np.array(points)
        @self.cache_source_under(name)
        def point_collection_source_data(points, color, scale_factor):
            return mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color, scale_factor=scale_factor).mlab_source
        point_collection_source_data(points, color, scale_factor).reset(x=points[:, 0], y=points[:, 1], z=points[:, 2])

    def draw(self):
        # Environment grid
        self.plot_box("big box", 20)
        for i, vertex in enumerate(cube_vertices):
            self.plot_box(f"octant box {i}", 10, (np.array(vertex)-.5)*10)
            break
        
        # Plot the coordinate systems
        self.plot_bases()

        # Plot the rigid objects at their actual positions in the environment
        for rigidobject in self.environment.rigidobjects:
            self.plot_point_collection(f"rigidobject {rigidobject}", rigidobject.get_points())
            #self.plot_box(f"rigidobject {rigidobject} box", 1, rigidobject.origin, rigidobject.orientation)

        #import pdb; pdb.set_trace()

        # Example for testing the ellipsoid

        if False:
            focal_distance = 3
            depth_of_field = 1
            sensor_size = (np.pi/2, np.pi/4)
            resolution = (100, 50)
            img_space_angles = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
            # scale img_space_angles to actual angles using the camera sensor size. Also, the zero-point for phi is pi/2, not 0.
            angles = np.array(img_space_angles) * np.array(sensor_size) / 2 + np.array([0, np.pi/2])
            print(angles, "angles")
            # Compute means in camera space
            means_in_camera_cartesian_space = np.array([
                -np.tan(angles[:, 0]) * focal_distance,
                np.tan(angles[:, 1]-(np.pi/2)) * focal_distance,
                [-focal_distance] * len(angles)#-focal_distance * np.cos(angles[:, 0]) * np.sin(angles[:, 1])
            ]).T
            print(means_in_camera_cartesian_space, "means_in_camera_cartesian_space")
            self.plot_point_collection(means_in_camera_cartesian_space, color=(1, 0, 0))

            # Transform means to the coordinate system space
            means_coord_sys_space = means_in_camera_cartesian_space # null for now. previously inverse_transform_points(self.position, self.orientation, means_in_camera_cartesian_space)

            # Define spherical covariance matrix
            cov_spherical = np.diag([
                depth_of_field**2,  # Large variance in radial direction
                (sensor_size[0] / resolution[0])**2,  # Small variance in angular theta direction
                (sensor_size[1] / resolution[1])**2  # Small variance in angular phi direction
            ])

            # List to collect Cartesian covariances
            cov_cartesian_list = []

            # Convert each point's covariance from spherical to Cartesian
            for theta, phi in angles:
            # Calculate the Jacobian matrix J for the transformation
                J = np.array([
                    [np.sin(phi) * np.sin(theta), focal_distance * np.sin(phi) * np.cos(theta), focal_distance * np.cos(phi) * np.sin(theta)],
                    [np.cos(phi), 0, -focal_distance * np.sin(phi)],
                    [np.sin(phi) * np.cos(theta), -focal_distance * np.sin(phi) * np.sin(theta), focal_distance * np.cos(phi) * np.cos(theta)]
                ])

                # Compute the Cartesian covariance matrix for the point
                cov_cartesian = J @ cov_spherical @ J.T
                print(cov_cartesian)
                cov_cartesian_list.append(cov_cartesian)

            for example_mean, example_covariance in zip(means_coord_sys_space, cov_cartesian_list):
                # Plot an ellipsoid at the test mean and covariance
                self.plot_covariance_ellipsoid(example_mean, example_covariance)

        # Plot the observers and make measurements. show the measurements as ellipsoids.
        for coord_sys, origin, orientation in self.aligner.iterate_coordinate_systems():
            for i, (mean, covariance) in enumerate(coord_sys.measurements):
                self.plot_covariance_ellipsoid(f"coordinate system {coord_sys} measurement {i} ellipsoid", transform_point(origin, orientation, mean), rotate_covariance(orientation, covariance), color=(0, 1, 1), opacity=0.2)
            for observer in coord_sys.observers:
                if isinstance(observer, PointTrackerObserver):
                    self.plot_box(f"tracker {observer} box", 1, transform_point(origin, orientation, observer.position), orientation=qprod(orientation, observer.orientation), color=(0, 1, 1))
                if isinstance(observer, CameraObserver):
                    #observer has observer.position, observer.orientation
                    #we need to place it relative to the coordinate system's origin and orientation, meaning that its position and orientation need to be modified before plotting
                    self.plot_camera(f"camera {observer}", transform_point(origin, orientation, observer.position), qprod(orientation, observer.orientation))
                    self.plot_point_collection(f"camera corners {observer}", transform_points(origin, orientation, observer.image_corners()), color=(1, 0, 0), scale_factor=0.1)
                #     for rigidobject in self.environment.rigidobjects:
                #         img_space_angles = self.environment.project_to_image(observer, rigidobject.get_points())
                #         # plot rays (lines) from the camera to the points in coordinate space
                #         end_points = observer.image_space_angles_to_coord_space(img_space_angles)
                #         start_point = transform_point(origin, orientation, observer.position)
                #         for end_point in end_points:
                #             mlab.plot3d([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=(1, 0, 0), tube_radius=None, line_width=1)
                #         measurements = observer.measure(img_space_angles)
                #         for mean, covariance in measurements:
                #             self.plot_covariance_ellipsoid(mean, covariance)

    def show(self, update_fn=lambda:None, delay=100):
        print("called 'show()'")
        self.draw()

        @mlab.animate(delay=delay)
        def animate():
            while True:
                with RenderOff():
                    update_fn()
                    self.draw()
                yield

        _ = animate() # store in a variable to keep the animation running
        mlab.show()
    
    def show_and_save_gif(self, filename, update_fn=lambda: None, num_frames=100):
        import imageio
        print("called 'save_gif()'")
        writer = imageio.get_writer(filename, mode='I', duration=10)

        def saving_update_fn():
            update_fn()
            img = mlab.screenshot(antialiased=True)
            writer.append_data(img)

        try:
            writer = imageio.get_writer(filename, mode='I', duration=10)
            self.show(update_fn=saving_update_fn)
        finally:
            writer.close()
            print("GIF saved as", filename)


    def close(self):
        self.closed = True
        mlab.close()
