from nestbox.coordsystem import PointTrackerObserver
from nestbox.aligner import AdamAligner, GradientAligner
from nestbox.sim import SimEnvironment, RigidObject
from nestbox.run_optimizer import run_optimizer
from nestbox.measurement import NormalMeasurement
from nestbox.feature import to_feature
from visualizer import Visualizer
import numpy as np
import pyquaternion
import sys
import redis
import json
import threading
from nestbox.test_aligner import init_random_coordinate_system
from nestbox.test_conn import receive_full_message
import socket
import sys
import time
from nestbox.protos import Twig

def init_random_tracker_coordinate_system():
    cs, origin, orientation = init_random_coordinate_system()
    #origin = (0, 0, 0)
    cs.add_local_observer(PointTrackerObserver(variance=.000003, position=(np.random.rand(3)-.5)*.5, orientation=pyquaternion.Quaternion.random()))
    return cs, origin, orientation

hand_points = [
    (0.49591675, 1.04020393, 0.09741183),
    (0.49839216, 1.06511402, 0.0956339),
    (0.50531477, 1.08689547, 0.08672391),
    (0.51517296, 1.11415505, 0.07295427),
    (0.52733737, 1.14382184, 0.06376299),
    (0.53618872, 1.12620354, 0.12092253),
    (0.55311155, 1.15736365, 0.13292672),
    (0.56270504, 1.17794931, 0.14065033),
    (0.53118712, 1.11668336, 0.14017671),
    (0.54086423, 1.15039694, 0.16394688),
    (0.54556179, 1.17263532, 0.17887434),
    (0.51756322, 1.10737383, 0.1523881),
    (0.52353334, 1.13472342, 0.1788058),
    (0.52646852, 1.15398145, 0.19636935),
    (0.49205095, 1.06454217, 0.13096154),
    (0.50051725, 1.09720945, 0.16081931),
    (0.49868798, 1.11353385, 0.18630677),
    (0.49543947, 1.12399328, 0.20309892),
    (0.53778297, 1.15961957, 0.04853401),
    (0.5722397, 1.19584823, 0.14942937),
    (0.5515421, 1.19156182, 0.19352266),
    (0.52982402, 1.17220891, 0.21171884),
    (0.49422598, 1.13624716, 0.22093384),
]
# center points at origin
hand_points = (np.array(hand_points) - np.mean(hand_points, axis=0))*10



def hand_rigid_object(mirror=False):
    origin = np.array((1, 0, 0))
    quaternion = pyquaternion.Quaternion.random()
    rigid_object = RigidObject(origin, quaternion)
    points = np.array(hand_points)
    if mirror:
        points[:, 0] *= -1
        origin *= -1
    rigid_object.add_points(points)
    return rigid_object

if __name__ == "__main__":

    environment = SimEnvironment()

    # Create an aligner and add some random coordinate systems
    # aligner = GradientAligner()
    aligner = AdamAligner()

    base_coord_sys, base_origin, base_orientation = init_random_tracker_coordinate_system()
    aligner.add_coordinate_system(base_coord_sys, base_origin, base_orientation)
    # init dict that maps stream ids to coordinate systems
    coordinate_systems_for_streams = {}
    rigidobjects_for_streams = {}
    for mirror, streamid in [(True, 'lefthand'), (False, 'righthand')]:
        hand_rigid = hand_rigid_object(mirror)
        environment.add_rigidobject(hand_rigid)
        hand_coord_sys, hand_coord_sys_origin, hand_coord_sys_orientation = init_random_tracker_coordinate_system()
        aligner.add_coordinate_system(hand_coord_sys, hand_coord_sys_origin, hand_coord_sys_orientation)
        coordinate_systems_for_streams[streamid] = hand_coord_sys
        rigidobjects_for_streams[streamid] = hand_rigid

    aligner.pin(base_coord_sys)
    stream_for_coord_sys = {v: k for k, v in coordinate_systems_for_streams.items()}
    stream_for_rigidobject = {v: k for k, v in rigidobjects_for_streams.items()}


    for coord_sys, origin, orientation in aligner.iterate_coordinate_systems():
        environment.place_coordinate_system(coord_sys, (0, 0, 0))
        for obs in coord_sys.observers:
            for rigidobject in environment.rigidobjects:
                if coord_sys is base_coord_sys or rigidobject is rigidobjects_for_streams[stream_for_coord_sys[coord_sys]]:
                    print(f"added points from {obs} to {coord_sys} (base cs is {base_coord_sys})")
                    points = rigidobject.get_points()
                    feature_ids = [stream_for_rigidobject[rigidobject] + f'_{i}' for i in range(len(points))]
                    points_dict = {feature_ids[i]: point for i, point in enumerate(points)}
                    if isinstance(obs, PointTrackerObserver):
                        observer_points = environment.points_from_observer_perspective(obs, points) + np.random.normal(0, obs.variance**.5*.1, (len(points), 3)) #TODO
                        measurements = obs.measure({feature_ids[i]: obs_point for i, obs_point in enumerate(observer_points)})
                        coord_sys.update_measurements(measurements)

    if "--visualize-graph" in sys.argv:
        aligner.build_model(visualization=True)
        exit()

    # Connect to Redis

    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    # check if connected
    try:
        redis_client.ping()
    except redis.exceptions.ConnectionError:
        print("Redis server not running. Start Redis server with 'redis-server'.")

    # Function to publish optimization updates
    def publish_updates(channel, state):
        # state: a dictionary containing the current state of the optimization
        try:
            redis_client.publish(channel, json.dumps(state))
        except redis.exceptions.ConnectionError:
            pass

    def redis_listener():
        pubsub = redis_client.pubsub()
        pubsub.subscribe(['optimization_update', 'pin_command'])
        print("Redis listener started")

        for message in pubsub.listen():
            if message['type'] == 'message':
                if message['channel'].decode('utf-8') == 'pin_command':
                    pin_data = json.loads(message['data'])
                    pin_index = pin_data['pin']
                    print(f"Received pin command for coordinate system {pin_index}")
                    aligner.pin(pin_index)

    # Start the listener in a separate thread
    threading.Thread(target=redis_listener, daemon=True).start()

    live_hand_point_map = {to_feature(streamid + f'_{i}'): point
                           for streamid, rigidobject in rigidobjects_for_streams.items()
                           for i, point in enumerate(rigidobject.get_points())}

    # Visualizer
    visualizer = Visualizer(aligner, environment)

    def callback(aligner):
        for feature_id, meas in base_coord_sys.measurements.items():
            if feature_id in live_hand_point_map:
                if not isinstance(meas, NormalMeasurement):
                    continue
                base_coord_sys.measurements[feature_id].mean = live_hand_point_map[feature_id]
        # Send optimization state to Redis
        visualizer.draw()
        state = visualizer.state()
        publish_updates('optimization_update', state)
    
    threading.Thread(target=run_optimizer, args=(aligner, callback), daemon=True).start()

    # while True:
    #     time.sleep(1)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ip address from cmd args with localhost default
    ip_address = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    server_socket.bind((ip_address, 12345))
    print(f"Server is bound to {server_socket.getsockname()}")
    server_socket.listen(1)
    print("Server is listening...")
    connection, addr = server_socket.accept()
    print(f"Connected by {addr}")

    average_center = np.array([0, 0, 0])

    try:
        while True:
            data = receive_full_message(connection)
            if data is None:
                break
            try:
                twig = Twig(data)
                print("Received Twig")
                ms = twig.measurement_sets[0]
                new_hand_points = ms.means*10
                # skip if any of them are zero or approximately zero
                if np.any(np.abs(new_hand_points) < 1e-6):
                    continue
                average_center = .9*average_center + .1*np.mean(new_hand_points, axis=0)
                for i, point in enumerate(new_hand_points):
                    live_hand_point_map[to_feature(twig.stream_id + f'_{i}')] = point - average_center

            except Exception as e:
                # full traceback
                import traceback
                traceback.print_exc()
                print(f"Error: {e}")
    finally:
        connection.close()
