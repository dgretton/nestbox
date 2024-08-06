import numpy as np
from nestbox.api import OptimizedTransformer, Dim
from nestbox.numutil import coerce_numpy

def test_matrix_examples():
    # in_vec = np.array([150.0, 1.0, 2.0, 3.0, 1, 1, 1])
    # expected_out_vec = np.array([1.0, 2.0, 3.0, 150.0, 1, 2, 3])
    # test_mtx = np.array([
    #     [0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 2, 0],
    #     [0, 0, 0, 0, 0, 0, 3]
    # ])
    # out_vec = test_mtx @ in_vec
    # print(out_vec)
    # assert np.allclose(out_vec, expected_out_vec)
    # print("=== Test for manually constructed matrix passed ===")

    # from nestbox.aligner import AlignmentResult
    # test_align_result = AlignmentResult('', 0, [0, 0, 0], [1, 0, 0, 0], [1, 2, 3])
    # transformer = OptimizedTransformer([Dim.T, Dim.X, Dim.Y, Dim.Z], test_align_result, {Dim.VX: 1.0, Dim.VY: 2.0, Dim.VZ: 3.0})
    # test_mtx_2 = transformer._create_permutation_matrix()
    # print(test_mtx_2)
    # out_vec = test_mtx_2 @ in_vec
    # assert np.allclose(out_vec, expected_out_vec)
    # print("=== Test for OptimizedTransformer constructed matrix passed ===")
    in_vec = np.array([150.0, 1.0, 2.0, 3.0])
    expected_out_vec = np.array([1.0, 2.0, 3.0, 150.0])
    test_reord_mtx = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    out_vec = test_reord_mtx @ in_vec
    print(out_vec)
    assert np.allclose(out_vec, expected_out_vec)
    print("=== Test for manually constructed matrix passed ===")

    from nestbox.aligner import AlignmentResult
    q = np.array([0, 0, 0, 1])
    o = np.array([5, 0, -5])
    test_align_result = AlignmentResult('', 0, o, q, [0, 0, 0])
    transformer = OptimizedTransformer([Dim.T, Dim.X, Dim.Y, Dim.Z], test_align_result)
    test_reord_mtx_2 = transformer._permutation_matrix
    print(test_reord_mtx_2)
    assert np.allclose(test_reord_mtx_2, test_reord_mtx)
    print("=== Test for OptimizedTransformer constructed reorder matrix passed ===")
    test_quaternion = transformer.quaternion
    assert np.allclose(coerce_numpy(test_quaternion), coerce_numpy(q))
    test_rotation_mtx3d = test_quaternion.rotation_matrix
    print(test_rotation_mtx3d)
    assert np.allclose(test_rotation_mtx3d, np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]))
    print("=== Test for OptimizedTransformer quaternion and 3d rotation matrix passed ===")
    test_rot_mtx = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    assert np.allclose(transformer._rotation_matrix, test_rot_mtx)
    print("=== Test for OptimizedTransformer constructed rotation matrix passed ===")
    expected_transformed_out_vec = np.array([150.0, 4.0, -2.0, -2.0]) # [150.0, 1.0, 2.0, 3.0] rotated by quaternion [0, 0, 0, 1] and translated by [5, 0, -5]
    out_vec = transformer.transform(in_vec)
    print(out_vec)
    assert np.allclose(out_vec, expected_transformed_out_vec)
    print("=== Test for OptimizedTransformer transform passed ===")
    transformer = OptimizedTransformer([Dim.X, Dim.Y, Dim.Z], test_align_result, {Dim.T: 150.0})
    expected_transformed_out_vec = np.array([4.0, -2.0, -2.0]) # [1.0, 2.0, 3.0] rotated by quaternion [0, 0, 0, 1] and translated by [5, 0, -5]
    out_vec = transformer.transform([1, 2, 3])
    print(out_vec)
    assert np.allclose(out_vec, expected_transformed_out_vec)
    print("=== Test for OptimizedTransformer transform passed ===")
    transformer = OptimizedTransformer([Dim.X, Dim.Y, Dim.Z, Dim.T], test_align_result)
    expected_transformed_out_vec = np.array([4.0, -2.0, -2.0, 0.0]) # [1.0, 2.0, 3.0, t=0] rotated by quaternion [0, 0, 0, 1] and translated by [5, 0, -5]
    out_vec = transformer.transform([1, 2, 3, 0])
    print(out_vec)
    assert np.allclose(out_vec, expected_transformed_out_vec)
    real_response_str = """
    {
        "timestamp": "1722712501.0014799",
        "status": 0,
        "origin": [
            0.0018406854942440987,
            5.506013985723257e-05,
            0.0007134964689612389
        ],
        "quaternion": [
            0.0007226830760873049,
            0.41724294561834185,
            -0.8543152371008798,
            -0.3099242680391754
        ],
        "delta_velocity": [
            0.0,
            0.0,
            0.0
        ]
    }"""
    import json
    test_align_result = AlignmentResult.from_json(json.loads(real_response_str))
    transformer = OptimizedTransformer([Dim.X, Dim.Y, Dim.Z], test_align_result, {Dim.T: 150.0})
    print(transformer.transform([1, 2, 3]))
    print(transformer.transform_many([[1, 2, 3]]))

if __name__ == '__main__':
    test_matrix_examples()
    print('All tests passed.')