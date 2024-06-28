import torch
import numpy as np
from torch.autograd import profiler
from ..numutil import transform_point, rotate_covariance, coerce_numpy
from .aligner import Aligner
from ..coordsystem import NormalMeasurement

class TorchAligner(Aligner):

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

    def inverse_transform_point(self, origin, orientation, point):
        # Ensure point is tensor, if not already
        point = coerce_numpy(point)
        point = torch.tensor(point, dtype=torch.float32, requires_grad=True) if not isinstance(point, torch.Tensor) else point

        # Form the quaternion-like tensor by adding a zero scalar part
        point_quaternion = torch.cat((torch.tensor([0.0], requires_grad=True, dtype=torch.float32), point))

        translated_point = point_quaternion - torch.cat((torch.tensor([0.0], dtype=torch.float32), origin))
        #rotate by the inverse of the orientation
        rotated_point = self.hamilton_product(self.quaternion_conjugate(orientation), translated_point)
        rotated_point = self.hamilton_product(rotated_point, orientation)

        return rotated_point[1:]

    def quaternion_conjugate(self, q):
        """Returns the conjugate of the quaternion."""
        # Negative sign on the vector part, keep scalar part the same
        return torch.cat((q[0:1], -q[1:]))

    def hamilton_product(self, q1, q2):
        scalar = q1[0] * q2[0] - torch.sum(q1[1:] * q2[1:], dtype=torch.float32)
        vector = q1[0] * q2[1:] + q2[0] * q1[1:] + torch.cross(q1[1:], q2[1:])
        return torch.cat((scalar.unsqueeze(0), vector))

    def quaternion_distance(self, q1, q2):
        # Ensure quaternions are normalized
        q1 = q1 / torch.linalg.norm(q1)
        q2 = q2 / torch.linalg.norm(q2)
        # Calculate the cosine of the angle between the two quaternions
        cos_theta = torch.dot(q1, q2).abs()
        # Convert cosine into angular distance
        theta = torch.acos(cos_theta) * 2
        return theta

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

    # def generalized_jensen_shannon_divergence(self, feature_means, feature_covariances): TODO may come back to this
        # \mu_M = \frac{1}{N} \sum_{i=1}^N \mu_{P_i}
        # \Sigma_M = \frac{1}{N} \sum_{i=1}^N \Sigma_{P_i}
        # D_{KL}(P_i \| M) = \frac{1}{2} \left( \text{tr}(\Sigma_M^{-1} \Sigma_{P_i}) + (\mu_{P_i} - \mu_M)^T \Sigma_M^{-1} (\mu_{P_i} - \mu_M) - k + \ln\left(\frac{\det \Sigma_M}{\det \Sigma_{P_i}}\right) \right)
        # mixture_mean = torch.mean(feature_means, dim=0)
        # mixture_covariance = torch.mean(feature_covariances, dim=0)
        # mix_cov_inv = torch.linalg.inv(mixture_covariance)
        # mix_cov_det = torch.linalg.det(mixture_covariance)
        # divergence_terms = []
        # for mean, covariance in zip(feature_means, feature_covariances):
        #     mean_diff = mean - mixture_mean
        #     divergence_terms.append(0.5 * (torch.trace(mix_cov_inv @ covariance) + mean_diff @ mix_cov_inv @ mean_diff - mean.size(0) + torch.log(mix_cov_det / torch.linalg.det(covariance)))) #TODO: check lol


class GradientAligner(TorchAligner):

    def build_model(self, visualization=False):

        #if not self.stale(): TODO put back probably, once we know how to reset gradients without rebuilding the model
        #    return

        # print('BUILDING MODEL')

        self.origins = torch.tensor(coerce_numpy(self.current_origins), dtype=torch.float32, requires_grad=True)
        self.orientations = torch.tensor(coerce_numpy(self.current_orientations), dtype=torch.float32, requires_grad=True)

        # sampling stage
        # pick a random coordinate system
        # for every measurement in that coordinate system, sample a point from its distribution

        random_idx = np.random.randint(0, len(self.coordinate_systems))
        chosen_coord_sys = self.coordinate_systems[random_idx]

        all_other_feature_ids = set(k for cs in self.coordinate_systems if cs is not chosen_coord_sys for k in cs.measurements.keys())

        temp_sampled_points = []
        temp_sampled_features = []

        for feature_id, meas in chosen_coord_sys.measurements.items():
            if feature_id not in all_other_feature_ids:
                continue
            if not isinstance(meas, NormalMeasurement):
                raise ValueError("All measurements must be of type NormalMeasurement at the moment")
            chosen_mean, chosen_cov = meas.get_sample()
            chosen_mean = torch.tensor(transform_point(self.origins[random_idx], self.orientations[random_idx], chosen_mean), dtype=torch.float32)
            chosen_cov = torch.tensor(rotate_covariance(self.orientations[random_idx], chosen_cov), dtype=torch.float32)
            temp_sampled_points.append(torch.distributions.MultivariateNormal(chosen_mean, chosen_cov).sample())
            temp_sampled_features.append(feature_id)

        temp_sampled_points = torch.stack(temp_sampled_points)
        self.sampled_features = temp_sampled_features[:] # save for inspection
        self.sampled_points = temp_sampled_points.detach().numpy() # save for inspection
        self.sampled_cs = random_idx

        coord_sys_log_likelihoods = []

        for i, coord_sys in enumerate(self.coordinate_systems):
            #if i == random_idx:
            #    continue
            # all of these steps should happen in the pytorch framework so that we can compute gradients later
            # 1. transform the means of the measurements from coordinate system space to the "global space" where all the coordinate systems live.
            # 2. rotate the covariances of the measurements from coordinate system space to the "global space" using a conversion to a matrix and a pair of matrix multiplications (see rotate_covariance)
            # 3. compute the negative log likelihood of the current state of the system given the measurement means and covariances
            # 4. sum the negative log likelihoods of all the measurements
            # 5. store the sum in self.loss

            origin = self.origins[i]
            orientation = self.orientations[i]
            # for (mean, covariance), temp_known_point in zip(coord_sys.measurements, temp_sampled_points):
            #     mean = torch.tensor(mean, dtype=torch.float32, requires_grad=True)
            #     covariance = torch.tensor(covariance, dtype=torch.float32, requires_grad=True)
            #     global_space_mean = self.transform_point(origin, orientation, mean)
            #     global_space_covariance = self.rotate_covariance(orientation, covariance)
            #     coord_sys_log_likelihoods.append(self.multivariate_gaussian_log_likelihood(temp_known_point, global_space_mean, global_space_covariance))
            for j, feature_id in enumerate(temp_sampled_features):
                if feature_id not in coord_sys.measurements:
                    continue
                mean, covariance = coord_sys.measurements[feature_id].get_sample()
                mean = torch.tensor(mean, dtype=torch.float32, requires_grad=True)
                covariance = torch.tensor(covariance, dtype=torch.float32, requires_grad=True)
                global_space_mean = self.transform_point(origin, orientation, mean)
                global_space_covariance = self.rotate_covariance(orientation, covariance)
                coord_sys_log_likelihoods.append(self.multivariate_gaussian_log_likelihood(temp_sampled_points[j], global_space_mean, global_space_covariance))

        self.loss = -torch.sum(torch.stack(coord_sys_log_likelihoods), dtype=torch.float32)
        self.loss.retain_grad()

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

    def gradient_descent_step(self, learning_rate=0.000001):

        ### Build the model
        profile = False
        if profile:
            with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                with profiler.record_function("build_model"):
                    self.build_model()
                with profiler.record_function("model_backward"):
                    self.loss.backward()
        else:
            self.build_model()
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
            # If the gradients are larger than 1/learning_rate, scale them down
            max_grad = max(torch.abs(self.origins.grad).max(), torch.abs(self.orientations.grad).max())
            if max_grad > 1/learning_rate:
                self.origins.grad /= max_grad * learning_rate
                self.orientations.grad /= max_grad * learning_rate
            self.origins.grad *= self.learning_rate_factor
            self.orientations.grad *= self.learning_rate_factor
            self.origins -= learning_rate * self.origins.grad
            self.orientations -= learning_rate * self.orientations.grad * .01
            self.origins.grad.zero_()
            self.orientations.grad.zero_()
            # Renormalizing the orientations to maintain them as unit quaternions
            self.orientations /= torch.linalg.norm(self.orientations, dim=1, keepdim=True, dtype=torch.float32)
            if self.pinned_cs_idx is None:
                # Move the mean of the origins back to zero for stability
                self.origins -= torch.mean(self.origins, dim=0, keepdim=True)
            else:
                # Transform all coordinate systems so that the one at the specified index is at 0, 0, 0, and 1, 0, 0, 0
                pinned_origin = self.origins[self.pinned_cs_idx].clone()
                pinned_orientation = self.orientations[self.pinned_cs_idx].clone()
                for i in range(len(self.coordinate_systems)):
                    self.origins[i] = self.inverse_transform_point(pinned_origin, pinned_orientation, self.origins[i])
                    self.orientations[i] = self.hamilton_product(self.quaternion_conjugate(pinned_orientation), self.orientations[i])
                # assert that the one at the specified index is at 0, 0, 0, and 1, 0, 0, 0 (or all close)
                assert torch.allclose(self.origins[self.pinned_cs_idx], torch.tensor([0., 0., 0.]))
                assert torch.allclose(self.orientations[self.pinned_cs_idx], torch.tensor([1., 0., 0., 0.]))

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


class AdamAligner(GradientAligner):
    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = None  # First moment vector
        self.v_t = None  # Second moment vector
        self.t = 0       # Timestep

    def build_model(self, *args, **kwargs):
        super().build_model(*args, **kwargs)
        if self.m_t is None:
            self.m_t = {'origins': torch.zeros_like(self.origins), 'orientations': torch.zeros_like(self.orientations)}
        if self.v_t is None:
            self.v_t = {'origins': torch.zeros_like(self.origins), 'orientations': torch.zeros_like(self.orientations)}

    def gradient_descent_step(self, learning_rate=0.1):
        self.build_model()
        self.loss.backward()

        self.t += 1
        with torch.no_grad():
            for param_name in ['origins', 'orientations']:
                param = getattr(self, param_name)
                grad = param.grad #* (.0001 if param_name == 'orientations' else 1.0)

                # Update biased first moment estimate
                self.m_t[param_name] = self.beta1 * self.m_t[param_name] + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                self.v_t[param_name] = self.beta2 * self.v_t[param_name] + (1 - self.beta2) * (grad ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m_t[param_name] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v_t[param_name] / (1 - self.beta2 ** self.t)

                # Update parameters
                param -= learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                param.grad.zero_()

            # Renormalize orientations if necessary
            self.orientations /= torch.linalg.norm(self.orientations, dim=1, keepdim=True, dtype=torch.float32)
            if self.pinned_cs_idx is None:
                self.origins -= torch.mean(self.origins, dim=0, keepdim=True)
            else:
                pinned_origin = self.origins[self.pinned_cs_idx].clone()
                pinned_orientation = self.orientations[self.pinned_cs_idx].clone()
                for i in range(len(self.coordinate_systems)):
                    self.origins[i] = self.inverse_transform_point(pinned_origin, pinned_orientation, self.origins[i])
                    self.orientations[i] = self.hamilton_product(self.quaternion_conjugate(pinned_orientation), self.orientations[i])
                assert torch.allclose(self.origins[self.pinned_cs_idx], torch.tensor([0., 0., 0.]))
                assert torch.allclose(self.orientations[self.pinned_cs_idx], torch.tensor([1., 0., 0., 0.]))

        self.current_origins = [origin.detach().numpy() for origin in self.origins]
        self.current_orientations = [orientation.detach().numpy() for orientation in self.orientations]
