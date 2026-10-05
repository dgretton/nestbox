# run an aligner passed from an external source, presumably already set up with coordinate systems and observers
import time

def update(aligner):
    aligner.gradient_descent_step()

def run_optimizer(aligner, callback=None, interval=.066):
    start_time = time.time()
    while True:
        update(aligner)
        # aligners whose dynamics are defined in wall-clock time (GaussNewtonAligner) don't need to spin flat out
        time.sleep(getattr(aligner, 'step_interval', 0))
        if callback is not None and time.time() - start_time > interval:
            callback(aligner)
            start_time = time.time()
