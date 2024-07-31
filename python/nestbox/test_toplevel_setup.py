import nestbox

nestbox.create_coordinate_system('cs1')
nestbox.create_coordinate_system('cs2')
nestbox.add_measurement(
    feature='nestbox:feature/corner/0',
    cs='cs1',
    mean=[1.0, 2.0, 3.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/1',
    cs='cs1',
    mean=[4.0, 5.0, 6.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/2',
    cs='cs1',
    mean=[7.0, 8.0, -9.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/0',
    cs='cs2',
    mean=[-1.0, -2.0, 3.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/1',
    cs='cs2',
    mean=[-4.0, -5.0, 6.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.add_measurement(
    feature='nestbox:feature/corner/2',
    cs='cs2',
    mean=[-7.0, -8.0, -9.0],
    covariance=[[0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01]])
nestbox.start_alignment()