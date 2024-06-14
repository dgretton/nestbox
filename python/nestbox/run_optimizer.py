# run an aligner passed from an external source, presumably already set up with coordinate systems and observers
import time

def update(aligner):
    aligner.gradient_descent_step()

def run_optimizer(aligner, callback=None, interval=.05):
    start_time = time.time()
    while True:
        update(aligner)
        if callback is not None and time.time() - start_time > interval:
            callback(aligner)
            start_time = time.time()
