from coordsystem import CoordinateSystem, Aligner, CameraObserver, PointTrackerObserver, look_rotation, transform_point
from sim import SimEnvironment, RigidObject
from run_optimizer import run_optimizer
from visualizer import Visualizer
import numpy as np
import pyquaternion
import sys
import redis
import json

# function for a random coordinate system
def init_random_coordinate_system():
    initial_origin = (np.random.rand(3)-.5)*10
    initial_quaternion = pyquaternion.Quaternion.random()
    return CoordinateSystem(), initial_origin, initial_quaternion

def standard_camera(position=(0, 0, 0), sensor_size=(np.pi/2, np.pi/2), resolution=(300, 200), focal_distance=4, depth_of_field=2):
    return CameraObserver(position=position, sensor_size=sensor_size, resolution=resolution, focal_distance=focal_distance, depth_of_field=depth_of_field)

def init_random_binocular_coordinate_system():
    cs, initial_origin, initial_quaternion = init_random_coordinate_system()
    c1 = standard_camera(position=(1, 0, 0))
    c2 = standard_camera(position=(-1, 0, 0))
    # c3 = standard_camera(position=(0, 2, 0))
    # c4 = standard_camera(position=(0, -2, 0))
    cs.add_local_observer(c1)
    cs.add_local_observer(c2)
    # cs.add_local_observer(c3)
    # cs.add_local_observer(c4)
    #initial_quaternion = look_rotation(np.array(look_at_point) - c1.forward()) # help by pointing cameras at look_at_point
    return cs, initial_origin, initial_quaternion

def init_simple_binocular_coordinate_system():
    cs = CoordinateSystem()
    cs.add_local_observer(standard_camera(position=(1, 0, 0)))
    cs.add_local_observer(standard_camera(position=(-1, 0, 0)))
    return cs, (0, 0, -4), pyquaternion.Quaternion(1, 0, 0, 0)

def init_random_tracker_coordinate_system():
    cs, origin, orientation = init_random_coordinate_system()
    cs.add_local_observer(PointTrackerObserver(variance=.003))#, orientation=pyquaternion.Quaternion.random()))
    return cs, origin, orientation

def init_simple_tracker_coordinate_system():
    cs = CoordinateSystem()
    cs.add_local_observer(PointTrackerObserver(position=(.5, .5, .5), variance=.3))
    return cs, (0, 0, 0), pyquaternion.Quaternion(1, 0, 0, 0)

def random_rigid_object(pos_diam=20): # make a rigidobject with a random origin and orientation. then, given a mean and variance, sample a bunch of points and add them to the object
    origin = (np.random.rand(3)-.5) * pos_diam
    quaternion = pyquaternion.Quaternion.random()
    rigid_object = RigidObject(origin, quaternion)
    mean = np.random.rand(3)-.5
    std_dev = .5
    if camera_demo:
        num_points = 20
    if tracker_demo:
        num_points = 5
    rigid_object.add_points(np.random.normal(mean, std_dev, (num_points, 3)))
    return rigid_object

def origin_rigid_object(pos=(0, 0, 0)):
    rigid_object = RigidObject(pos)
    rigid_object.add_points([(0, 0, 0)])
    return rigid_object

if __name__ == "__main__":
    tracker_demo = True
    camera_demo = False
    if "--camera" in sys.argv:
        camera_demo = True
        tracker_demo = False
    simple = "--simple" in sys.argv

    environment = SimEnvironment()
    if simple:
        environment.add_rigidobject(origin_rigid_object())
        environment.add_rigidobject(origin_rigid_object((1, 0, 0)))
        environment.add_rigidobject(origin_rigid_object((0, 1, 0)))
        environment.add_rigidobject(origin_rigid_object((0, 0, 1)))
    else:
        if camera_demo:
            environment.add_rigidobject(random_rigid_object(0))
        if tracker_demo:
            for _ in range(3):
                environment.add_rigidobject(random_rigid_object())

    # Create an aligner and add some random coordinate systems
    aligner = Aligner()
    if simple:
        if camera_demo:
            aligner.add_coordinate_system(*init_simple_binocular_coordinate_system())
        if tracker_demo:
            aligner.add_coordinate_system(*init_simple_tracker_coordinate_system())
    else:
        if tracker_demo:
            aligner.add_coordinate_system(*init_random_tracker_coordinate_system())
            aligner.add_coordinate_system(*init_random_tracker_coordinate_system())
        if camera_demo:
            aligner.add_coordinate_system(*init_random_binocular_coordinate_system())
            aligner.add_coordinate_system(*init_random_binocular_coordinate_system())
        
    all_measured_points = []

    for coord_sys, origin, orientation in aligner.iterate_coordinate_systems():
        if simple:
            environment.place_coordinate_system(coord_sys, origin, orientation) # place at actual latent positions for now, i.e. the ground truth, with the problem already solved. purpose is to test stability and the visualization of uncertainties
        else:
            environment.place_coordinate_system(coord_sys, (0, 0, -4), (1, 0, 0, 0))
        for obs in coord_sys.observers:
            for rigidobject in environment.rigidobjects:
                points = rigidobject.get_points()
                if isinstance(obs, PointTrackerObserver):
                    for _ in range(3):
                        all_measured_points.extend(points)
                        obs.measure(environment.points_from_observer_perspective(obs, points) + np.random.normal(0, obs.variance**.5, (len(points), 3)))
                elif isinstance(obs, CameraObserver):
                    all_measured_points.extend(points)
                    obs.measure(environment.project_to_image(obs, points))

    if "--visualize-graph" in sys.argv:
        aligner.build_model(all_measured_points, visualization=True)
        exit()

    # Connect to Redis
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    # Function to publish optimization updates
    def publish_updates(channel, state):
        # state: a dictionary containing the current state of the optimization
        redis_client.publish(channel, json.dumps(state))

    # visualizer
    visualizer = Visualizer(aligner, environment)

    def callback(_):
        # Send optimization state to Redis
        visualizer.draw()
        state = visualizer.state()
        publish_updates('optimization_update', state)
    
    run_optimizer(aligner, all_measured_points, callback=callback)
