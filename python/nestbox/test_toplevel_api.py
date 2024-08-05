import json
import nestbox
from nestbox.sim import SimEnvironment, RigidObject
from nestbox.coordsystem import CoordinateSystem
from protos import MeasurementSet

# a truly awful workaround where I make a sim environment, a coordinate system, add observers to it, place it in the environment, and then measure points from the perspective of the observers, just so I can go grab the measurements lists from the coordinate systems and send them over the api
# I'm not proud of this <= honest to goodness suggested autocomplete by copilot omg O_O
# This would not have happened if I'd properly organized observers under the sim system, but I didn't want to change the existing code too much right now
def rigid_object():
    origin = (0, 0, 1)
    rigid_object = RigidObject(origin)
    rigid_object.add_points([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return rigid_object

environment = SimEnvironment()
environment.add_rigidobject(rigid_object())

cs1 = CoordinateSystem()
cs2 = CoordinateSystem()

environment.place_coordinate_system(cs1, (-1, 0, 0))
environment.place_coordinate_system(cs2, (1, 0, 0))

def get_measurements(coord_sys_name):
    coord_sys = {'cs1': cs1, 'cs2': cs2}[coord_sys_name]
    coord_sys.measurements = []
    for observer in coord_sys.observers:
        for rigidobject in environment.rigidobjects:
            points = rigidobject.get_points()
            observer.measure(environment.points_from_observer_perspective(observer, points))
    return coord_sys.measurements[:]

def get_measurement_set(coord_sys_name):
    measurements = get_measurements(coord_sys_name)
    print(measurements)
    exit()
    return MeasurementSet(samples=measurements)

def measurement_set_request(coord_sys_name):
    request = {'type': 'add_measurement_set',
               'cs_name': coord_sys_name}
    request.update(get_measurement_set(coord_sys_name).to_json())
    return request


if __name__ == '__main__':
    cs1_point = [2, 3, 4]
    cs2_point = nestbox.from_cs('cs1').to_cs('cs2').convert(cs1_point)
    print(f'Point in cs1: {cs1_point}. Point in cs2: {cs2_point}')
