import nestbox

nestbox.create_coordinate_system('cs1')
nestbox.create_coordinate_system('cs2')

# with offset
nestbox.add_measurement(
    feature='nestbox:feature/corner/0',
    cs='cs1',
    mean=[2.0, 3.0, 4.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/1',
    cs='cs1',
    mean=[5.0, 6.0, 7.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/2',
    cs='cs1',
    mean=[8.0, 9.0, -8.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])

# no offset
# nestbox.add_measurement(
#     feature='nestbox:feature/corner/0',
#     cs='cs1',
#     mean=[1.0, 2.0, 3.0],
#     covariance=[[0.001, 0, 0],
#                 [0, 0.001, 0],
#                 [0, 0, 0.001]])
# nestbox.add_measurement(
#     feature='nestbox:feature/corner/1',
#     cs='cs1',
#     mean=[4.0, 5.0, 6.0],
#     covariance=[[0.001, 0, 0],
#                 [0, 0.001, 0],
#                 [0, 0, 0.001]])
# nestbox.add_measurement(
#     feature='nestbox:feature/corner/2',
#     cs='cs1',
#     mean=[7.0, 8.0, -9.0],
#     covariance=[[0.001, 0, 0],
#                 [0, 0.001, 0],
#                 [0, 0, 0.001]])

nestbox.add_measurement(
    feature='nestbox:feature/corner/0',
    cs='cs2',
    mean=[-2.0, 1.0, 3.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/1',
    cs='cs2',
    mean=[-5.0, 4.0, 6.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/2',
    cs='cs2',
    mean=[-8.0, 7.0, -9.0],
    covariance=[[0.001, 0, 0],
                [0, 0.001, 0],
                [0, 0, 0.001]])

nestbox.start_alignment()
