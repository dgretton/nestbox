# run an aligner passed from an external source, presumably already set up with coordinate systems and observers
import time

def update(aligner, all_measured_points): # TODO remove all_measured_points
    aligner.gradient_descent_step(temp_known_points=all_measured_points)

def run_optimizer(aligner, all_measured_points, callback=None, interval=.2):
    start_time = time.time()
    while True:
        update(aligner, all_measured_points)
        if callback is not None and time.time() - start_time > interval:
            callback(aligner)
            for _, origin, orientation in aligner.iterate_coordinate_systems():
                print(f"current coordinate system position: {origin}")
                print(f"current coordinate system orientation: {orientation}")
            start_time = time.time()