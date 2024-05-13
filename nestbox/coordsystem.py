import pyquaternion
import torch
from torch.autograd import profiler
import numpy as np

def transform_point(origin, quaternion, point):
    origin = coerce_numpy(origin)
    quaternion = coerce_quaternion(quaternion)
    point = coerce_numpy(point)
    return coerce_numpy(quaternion.rotate(point) + np.array(origin))

def inverse_transform_point(origin, quaternion, point):
    origin = coerce_numpy(origin)
    quaternion = coerce_quaternion(quaternion)
    point = coerce_numpy(point)
    return coerce_numpy(quaternion.inverse.rotate(point - origin))

def transform_points(origin, quaternion, points):
    return np.array([transform_point(origin, quaternion, point) for point in points]) # TODO optimize

def inverse_transform_points(origin, quaternion, points):
    return np.array([inverse_transform_point(origin, quaternion, point) for point in points]) # TODO optimize

def rotate_covariance(quaternion, covariance):
    quaternion = coerce_quaternion(quaternion)
    covariance = coerce_numpy(covariance)
    return quaternion.rotation_matrix @ covariance @ quaternion.rotation_matrix.T

def quaternion_to_basis(quaternion):
    quaternion = coerce_quaternion(quaternion)
    return quaternion.rotate([1, 0, 0]), quaternion.rotate([0, 1, 0]), quaternion.rotate([0, 0, 1])

def look_rotation(forward, up=(0, 1, 0)):
    forward = coerce_numpy(forward)
    up = np.array(up)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation_matrix = np.array([right, up, -forward])
    
    # Check if the rotation matrix is special orthogonal
    if not np.isclose(np.linalg.det(rotation_matrix), 1.0):
        raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")
    
    quaternion = pyquaternion.Quaternion(matrix=rotation_matrix)
    return quaternion

def coerce_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float32:
            return tensor
        return tensor.astype(np.float32)
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    elif isinstance(tensor, pyquaternion.Quaternion):
        return np.array(list(tensor), dtype=np.float32)
    else:
        return np.array(tensor, dtype=np.float32)
    
def coerce_quaternion(tensor):
    if isinstance(tensor, np.ndarray):
        return pyquaternion.Quaternion(*tensor)
    elif isinstance(tensor, torch.Tensor):
        return pyquaternion.Quaternion(*tensor.detach().numpy())
    elif isinstance(tensor, pyquaternion.Quaternion):
        return tensor
    else:
        return pyquaternion.Quaternion(*tensor)


class CoordinateSystem:
    names = set()

    def __init__(self, name=None):
        self.observers = []
        self.measurements = [] # tuples of (mean, covariance) for each measurement, compiled from all observers
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
        observer.bind(self)

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
    
    def bind(self, coordinate_system):
        self.coordinate_system = coordinate_system
    
    def forward(self):
        return transform_point(self.position, self.orientation, [0, 0, 1])
    
    def add_measurements(self, measurement_means_and_covariances):
        self.coordinate_system.set_stale() # mark that the model will now need to be rebuilt before more optimization can happen
        self.coordinate_system.measurements.extend(measurement_means_and_covariances)

class PointTrackerObserver(Observer):
    def __init__(self, position=(0, 0, 0), orientation=pyquaternion.Quaternion(1, 0, 0, 0), variance=1.0):
        super().__init__(position, orientation)
        self.variance = variance

    def measure(self, points):
        means = transform_points(self.position, self.orientation, points)
        covariances = [coerce_numpy(np.eye(3) * self.variance)] * len(points) # coerce is for data type
        self.add_measurements(zip(means, covariances))
        return zip(means, covariances)

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
    
    def measure(self, img_space_angles):
        # scale img_space_angles to actual angles using the camera sensor size.  Also, the zero-point for phi is pi/2, not 0, to avoid pole singularity at phi=0.
        print(img_space_angles, "img_space_angles")
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
        self.add_measurements(zip(means_coord_sys_space, covs_coord_sys_space))
        return zip(means_coord_sys_space, covs_coord_sys_space)
    
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
        

class Aligner:
    def __init__(self):
        self.coordinate_systems = []
        # name everything so it can show up in the computation graph
        self.current_origins = []
        self.current_orientations = []
        self.loss = None # negative log likelihood output of the model with respect to which we will compute gradients, created in build_model
        self.learning_rate_factor = 1.0
        self.losses = []

    def add_coordinate_system(self, coord_sys, initial_origin=None, initial_orientation=None):
        if initial_origin is None:
            initial_origin = np.zeros(3)
        if initial_orientation is None:
            initial_orientation = np.array([1, 0, 0, 0])
        self.current_origins.append(coerce_numpy(initial_origin))
        self.current_orientations.append(coerce_numpy(initial_orientation))
        self.coordinate_systems.append(coord_sys)

    def reset_coordinate_system(self, coord_sys, set_origin, set_orientation):
        for i, cs in enumerate(self.coordinate_systems):
            if cs == coord_sys:
                self.current_origins[i] = coerce_numpy(set_origin)
                self.orientations[i] = coerce_numpy(set_orientation)
                coord_sys.set_stale(True)
                break
        raise ValueError("Coordinate system not found in aligner")

    def iterate_coordinate_systems(self):
        for i, coord_sys in enumerate(self.coordinate_systems):
            yield coord_sys, self.current_origins[i], self.current_orientations[i]

    def stale(self):
        for coord_sys in self.coordinate_systems:
            if coord_sys.stale:
                return True
        return False

    def build_model(self, temp_known_points, visualization=False):
        # temp_known_points is a list of points that are known to be in the global coordinate system space shared by the observers. Eventually, this will be replaced by another system to estimate the assignment of points to measurements.
        #if not self.stale(): TODO put back probably, once we know how to reset gradients without rebuilding the model
        #    return
        
        # print('BUILDING MODEL')

        self.origins = torch.tensor(coerce_numpy(self.current_origins), dtype=torch.float32, requires_grad=True)
        self.orientations = torch.tensor(coerce_numpy(self.current_orientations), dtype=torch.float32, requires_grad=True)

        coord_sys_log_likelihoods = []
        for i, coord_sys in enumerate(self.coordinate_systems):
            # all of these steps should happen in the pytorch framework so that we can compute gradients later
            # 1. transform the means of the measurements from coordinate system space to the "global space" where all the coordinate systems live.
            # 2. rotate the covariances of the measurements from coordinate system space to the "global space" using a conversion to a matrix and a pair of matrix multiplications (see rotate_covariance)
            # 3. compute the negative log likelihood of the current state of the system given the measurement means and covariances
            # 4. sum the negative log likelihoods of all the measurements
            # 5. store the sum in self.loss

            temp_known_points = torch.tensor(coerce_numpy(temp_known_points), dtype=torch.float32, requires_grad=True)
            origin = self.origins[i]
            orientation = self.orientations[i]
            for (mean, covariance), temp_known_point in zip(coord_sys.measurements, temp_known_points):
                mean = torch.tensor(mean, dtype=torch.float32, requires_grad=True)
                covariance = torch.tensor(covariance, dtype=torch.float32, requires_grad=True)
                global_space_mean = self.transform_point(origin, orientation, mean)
                global_space_covariance = self.rotate_covariance(orientation, covariance)
                coord_sys_log_likelihoods.append(self.multivariate_gaussian_log_likelihood(temp_known_point, global_space_mean, global_space_covariance))

        def print_grad(grad):
            print(grad)

        #self.origins.register_hook(print_grad)
        #self.orientations.register_hook(print_grad)
        self.loss = -torch.sum(torch.stack(coord_sys_log_likelihoods), dtype=torch.float32)
        self.loss.retain_grad()
        #self.loss.register_hook(print_grad)

        def print_grad_fn_chain(grad_fn, level=0):
            if grad_fn is None:
                return
            indent = ' ' * 4 * level
            print(f"{indent}{grad_fn}")
            if hasattr(grad_fn, 'variable'):  # Check if grad_fn is associated with a tensor
                tensor = grad_fn.variable
                print(f"  {indent}Gradient: {tensor.grad if tensor.requires_grad else 'No grad'}")
                tensor.register_hook(print_grad)
            for sub_fn, _ in grad_fn.next_functions:
                if sub_fn is not None:
                    print_grad_fn_chain(sub_fn, level+1)
        #print_grad_fn_chain(self.loss.grad_fn)

        for coord_sys in self.coordinate_systems:
            coord_sys.set_stale(False)
        
        if visualization:
            #render the computation graph
            import torchviz
            dot = torchviz.make_dot(self.loss)
            # show it
            dot.view()
            # save it to an image
            dot.render('model_graph', format='png')

    

    def gradient_descent_step(self, learning_rate=0.001, temp_known_points=[]):

        ### Build the model
        profile = False
        if profile:
            with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                with profiler.record_function("build_model"):
                    self.build_model(temp_known_points)
                with profiler.record_function("model_backward"):
                    self.loss.backward()
        else:
            self.build_model(temp_known_points)
            self.loss.backward()
        
        ### Hyperparameter fine-tuning during optimization
        # keep track of the last N losses
        N = 100
        if len(self.losses) >= N:
            self.losses.pop(0)
        self.losses.append(float(self.loss))
        if len(self.losses) == N:
            # see if the first 5 digits are the same. convert to scientific notation using format :e
            num_unique_losses = len(set(format(loss, '.5e')[:6] for loss in self.losses))
            is_oscillating = self.list_is_oscillating(self.losses)
            if (is_oscillating and num_unique_losses < N) or num_unique_losses < N - 3:
                print("Loss has entered a cycle, reducing learning rate.")
                self.learning_rate_factor *= .5
                print(f"new learning rate factor: {self.learning_rate_factor}")
                self.losses = []
            elif is_oscillating:
                print("Loss is oscillating, mildly reducing learning rate.")
                self.learning_rate_factor *= .8
                print(f"new learning rate factor: {self.learning_rate_factor}")
                self.losses = []

        ### Update parameters
        with torch.no_grad():
            # self.origins.grad /= self.grad_scale
            # self.orientations.grad /= self.grad_scale
            # If the gradients are larger than 1/learning_rate, scale them down
            max_grad = max(torch.abs(self.origins.grad).max(), torch.abs(self.orientations.grad).max())
            if max_grad > 1/learning_rate:
                self.origins.grad /= max_grad * learning_rate
                self.orientations.grad /= max_grad * learning_rate
            self.origins.grad *= self.learning_rate_factor
            self.orientations.grad *= self.learning_rate_factor
            # print(f"max origin grad: {torch.max(torch.abs(self.origins.grad))}, max orientation grad: {torch.max(torch.abs(self.orientations.grad))}")
            # print(f"loss: {self.loss}")
            self.origins -= learning_rate * self.origins.grad
            self.orientations -= learning_rate * self.orientations.grad * .01
            self.origins.grad.zero_()
            self.orientations.grad.zero_()
            # Renormalizing the orientations to maintain them as unit quaternions
            self.orientations /= torch.linalg.norm(self.orientations, dim=1, keepdim=True, dtype=torch.float32)

        # Detach the current states of origins and orientations
        self.current_origins = [origin.detach().numpy() for origin in self.origins]
        self.current_orientations = [orientation.detach().numpy() for orientation in self.orientations]

        if profile:
            # Print out the profiler results
            print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))
    
    def list_is_oscillating(self, l):
        up = l[1] > l[0]
        for i in range(2, len(l)):
            if up and l[i] > l[i-1]:
                return False
            elif not up and l[i] < l[i-1]:
                return False
            up = not up
        return True

    def transform_points(self, origin, orientation, points):
        return torch.stack([self.transform_point(origin, orientation, point) for point in points])
    
    def transform_point(self, origin, orientation, point):
        # Ensure point is tensor, if not already
        point = coerce_numpy(point)
        point = torch.tensor(point, dtype=torch.float32, requires_grad=True) if not isinstance(point, torch.Tensor) else point
        
        # Form the quaternion-like tensor by adding a zero scalar part
        point_quaternion = torch.cat((torch.tensor([0.0], requires_grad=True, dtype=torch.float32), point))  # Ensure this is a float tensor

        # Perform the Hamilton product for rotation (quaternion must be normalized)
        rotated_point = self.hamilton_product(orientation, point_quaternion)
        rotated_point = self.hamilton_product(rotated_point, self.quaternion_conjugate(orientation))

        # Adding the origin, ensure dimensions match
        result = rotated_point[1:] + origin  # Use the vector part of the quaternion

        return result

    def quaternion_conjugate(self, q):
        """Returns the conjugate of the quaternion."""
        # Negative sign on the vector part, keep scalar part the same
        return torch.cat((q[0:1], -q[1:]))

    def hamilton_product(self, q1, q2):
        scalar = q1[0] * q2[0] - torch.sum(q1[1:] * q2[1:], dtype=torch.float32)
        vector = q1[0] * q2[1:] + q2[0] * q1[1:] + torch.cross(q1[1:], q2[1:])
        return torch.cat((scalar.unsqueeze(0), vector))
    
    def quaternion_to_rotation_matrix(self, q):
        """ Convert a quaternion into a rotation matrix.
        
        Parameters:
            q (torch.Tensor): A tensor of shape (4,) representing the quaternion in the form [q0, q1, q2, q3]

        Returns:
            torch.Tensor: A 3x3 rotation matrix.
        """
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return torch.tensor([
            [1 - 2 * (q2 * q2 + q3 * q3),  2 * (q1 * q2 - q0 * q3),      2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3),      1 - 2 * (q1 * q1 + q3 * q3),  2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2),      2 * (q2 * q3 + q0 * q1),      1 - 2 * (q1 * q1 + q2 * q2)]
        ], dtype=torch.float32, requires_grad=True)
    
    def rotate_covariance(self, quaternion, covariance):
        return self.quaternion_to_rotation_matrix(quaternion) @ covariance @ self.quaternion_to_rotation_matrix(quaternion).T
    
    def multivariate_gaussian_log_likelihood(self, x, mean, covariance):
        """
        Calculate the multivariate Gaussian probability density function.

        Args:
        x: Tensor, the point at which to evaluate the PDF (shape: [d]).
        mean: Tensor, the mean vector of the Gaussian distribution (shape: [d]).
        covariance: Tensor, the covariance matrix of the Gaussian distribution (shape: [d, d]).

        Returns:
        pdf_value: Tensor, the value of the PDF evaluated at x.
        """
        k = mean.size(0)
        x_mean = x - mean
        cov_inv = torch.linalg.inv(covariance)
        det_cov = torch.linalg.det(covariance)
        normalization_constant_squared = (2 * torch.pi) ** k * det_cov
        exponent = -0.5 * x_mean @ cov_inv @ x_mean.t()
        log_pdf_value = exponent - torch.log(normalization_constant_squared) * 0.5 #TODO: rewrite to take advantage of log properties
        self.log_pdf_value = log_pdf_value
        return log_pdf_value


if __name__ == "__main__":
    a = Aligner()
    # test list_is_oscillating
    assert not a.list_is_oscillating([1, 2, 3, 4, 5, 4, 3, 2, 1])
    assert not a.list_is_oscillating([1, 2, -1, -2, -1, 2, 1])
    assert a.list_is_oscillating([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    assert a.list_is_oscillating([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    known_points = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
    means = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float32, requires_grad=True)
    covariances = [torch.eye(3) for _ in range(3)]
    coord_origins = torch.tensor([[0, 0, 0]], dtype=torch.float32, requires_grad=True)
    coord_orientations = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, requires_grad=True)
    a.origins = coord_origins
    a.orientations = coord_orientations
    transformed_means = a.transform_points(coord_origins[0], coord_orientations[0], means)
    transformed_covariances = [a.rotate_covariance(coord_orientations[0], cov) for cov in covariances]
    test_log_lik_grad = a.multivariate_gaussian_log_likelihood(known_points[0], transformed_means[0], transformed_covariances[0])

    class StaleCoordinateSystem(CoordinateSystem):
        def __init__(self):
            self.stale = True
            self.measurements = [(means[0], covariances[0])]
            self.observers = []
        def set_stale(self, stale):
            pass

    cs = StaleCoordinateSystem()
    #a.coordinate_systems = [cs]
    #pto = PointTrackerObserver(variance=0.3)
    #cs.add_local_observer(pto)
    a.add_coordinate_system(cs)
    #pto.measure([np.array([1, 1, 1])])
    
    a.build_model(known_points)
    #a.loss.backward()
    from torchviz import make_dot
    make_dot(test_log_lik_grad).render("test_computation_graph", format="png")
    #print(coord_orientations.grad[0])
    a.gradient_descent_step(learning_rate=0.01, temp_known_points=known_points)
