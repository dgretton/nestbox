"""Validation and actual JSON/Twig ingestion paths; no network or daemon process required."""
import contextlib
import io
import json
import unittest
from unittest.mock import Mock
import numpy as np

from nestbox.measurement import BearingMeasurement, NormalMeasurement, measurement_from_dict
from nestbox.feature import to_feature
from nestbox.protos import Dim, Twig, MeasurementSet
from nestbox.aligner.aligner_manager import AlignerManager
from nestbox.daemon_local_aligner_client import _DaemonLocalAlignerServer
from nestbox.coordsystem import CoordinateSystem
from nestbox.api_client import NestboxAPIClient
from nestbox.live_data_shim import LiveDataShim


class BearingPlumbingTests(unittest.TestCase):
    def test_cone_inspection_and_parallel_rays(self):
        from nestbox.aligner.gauss_newton_aligner import GaussNewtonAligner, _exp_so3, _matrix_to_quat
        from nestbox.aligner.measurement_residual import BearingResidual
        al = GaussNewtonAligner(convergence_time=0)
        cs = CoordinateSystem()
        f = to_feature('cone')
        m = BearingMeasurement(f, [0, 0, 1], [[.0004, .0001], [.0001, .0009]], origin=[.1, .2, .3])
        cs.measurements[f] = m
        R, t = _exp_so3(np.array([.2, .3, .4])), np.array([1., 2., 3.])
        al.add_coordinate_system(cs, t, _matrix_to_quat(R))
        _, feature, apex, axis, angles, length, basis = al.measurement_cones(k=2, include_basis=True)[0]
        self.assertEqual(feature, f)
        self.assertEqual(length, 2.)
        self.assertTrue(np.allclose(apex, R @ m.origin + t))
        self.assertTrue(np.allclose(axis, R @ m.direction))
        self.assertTrue(np.allclose(basis @ np.diag((angles/2)**2) @ basis.T,
                                    R @ m.basis @ m.angular_covariance @ m.basis.T @ R.T))
        al.feature_estimates[f] = (apex + 4*axis, np.eye(3))
        self.assertAlmostEqual(al.measurement_cones()[0][-1], 6.)
        al.gradient_descent_step()  # no shared features: stale inspection data must be cleared
        self.assertFalse(al.feature_estimates)
        self.assertEqual(al.measurement_cones()[0][-1], 2.)
        m.depth_prior = (3., .5)
        self.assertEqual(al.measurement_cones()[0][-1], 3.)
        ray = BearingResidual(BearingMeasurement(f, [0, 0, 1], .002))
        x = al._landmark([(0, ray), (1, ray)], [np.eye(3)]*2, [np.zeros(3)]*2)
        self.assertTrue(np.allclose(x, [0, 0, 1]))
        self.assertIsNone(al._landmark([(0, ray)], [np.eye(3)], [np.zeros(3)], members={0}))

    def test_validation_and_roundtrip(self):
        f = to_feature('bearing')
        m = BearingMeasurement(f, [0, 0, 5], .02, origin=[1, 2, 3], depth_prior=(2.5, 1), clear_key='camera')
        self.assertTrue(np.allclose(m.angular_covariance, np.eye(2)*.02**2))
        self.assertEqual(m.dimensions, [Dim.I, Dim.J, Dim.K])
        copy = measurement_from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(copy.to_dict(), m.to_dict())
        for kwargs in [dict(direction=[0, 0, 0]), dict(direction=[0, float('nan'), 1]),
                       dict(angular_covariance=-.1), dict(angular_covariance=[[1, 2], [2, 1]]),
                       dict(angular_covariance=[[1, .1], [0, 1]]), dict(origin=[1, 2]),
                       dict(basis=np.ones((3, 2))), dict(depth_prior=(0, 1)), dict(depth_prior=(1, 0))]:
            args = dict(direction=[0, 0, 1], angular_covariance=.02)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                BearingMeasurement(f, **args)

    def test_api_and_daemon_roundtrip(self):
        server = _DaemonLocalAlignerServer.__new__(_DaemonLocalAlignerServer)
        server.aligner_manager = AlignerManager({'type': 'gauss_newton'})
        cs = CoordinateSystem()
        server.aligner.add_coordinate_system(cs)
        client = NestboxAPIClient.__new__(NestboxAPIClient)
        client.base_url = 'http://local'
        client.session = Mock()
        def post(url, json):
            self.assertEqual(url, f'http://local/coordsys/{cs.name}/measurement')
            import json as codec
            response = codec.loads(server.process_data(codec.dumps(dict(
                type='add_measurements', request_id='test', cs_guid=cs.name, measurements=[json])).encode()))
            self.assertEqual(response['status'], 'success')
            return Mock(json=lambda: response)
        client.session.post.side_effect = post
        with contextlib.redirect_stdout(io.StringIO()):
            client.add_bearing_measurement('ray', cs.name, [1, 0, 2], .01, depth_prior=(2, .5), clear_key='cam')
            result = json.loads(server.process_data(json.dumps(dict(type='get_current_measurement', request_id='read',
                                              cs_guid=cs.name, feature_id='ray')).encode()))
        self.assertEqual(result['measurement']['type'], 'BearingMeasurement')
        self.assertEqual(result['measurement']['clear_key'], 'cam')
        self.assertTrue(np.allclose(result['measurement']['direction'], np.array([1, 0, 2])/np.sqrt(5)))
        # Exercise the public top-level export, not only the API client.
        import nestbox
        import nestbox.api as api
        from unittest.mock import patch
        with patch.object(api, '_get_client', return_value=client), contextlib.redirect_stdout(io.StringIO()):
            nestbox.add_bearing_measurement('ray', cs.name, [0, 0, 1], .01)

    def test_twig_router_shim_and_clear_keys(self):
        d = np.array([1., 2., 3.]); d /= np.linalg.norm(d)
        E = BearingMeasurement.tangent_basis(d)
        C = np.array([[.0004, .0001], [.0001, .0009]])
        covariance = E @ C @ E.T
        twig = Twig(stream_id='test_bearing_stream', coord_sys_id='unused')
        twig.measurement_sets = [MeasurementSet(samples=[(d, covariance)], dimensions=[Dim.I, Dim.J, Dim.K],
                                               is_homogeneous=[False]*3, transform=np.eye(3))]
        # Exercise protobuf serialization as well as the in-memory adapter.
        twig = Twig().load_bytes(twig.to_bytes())
        manager = AlignerManager({'type': 'gauss_newton'})
        cs = CoordinateSystem()
        manager.aligner.add_coordinate_system(cs)
        twig.coord_sys_id = cs.name
        manager.add_router(twig.stream_id, dict(stream_id=twig.stream_id, measurements=[dict(
            type='normal', feature_uri='ray', clear_key='camera', sample_pointer=dict(set=0, sample=0))]))
        untouched = to_feature('untouched')
        cs.measurements[untouched] = NormalMeasurement(untouched, [0, 0, 0], np.eye(3))
        with contextlib.redirect_stdout(io.StringIO()):
            manager.process_twig(twig)
            manager.process_twig(twig)  # replacement clears only this stream's measurements
        m = cs.measurements[to_feature('ray')]
        self.assertIsInstance(m, BearingMeasurement)
        self.assertTrue(np.allclose(m.basis @ m.angular_covariance @ m.basis.T, covariance))
        self.assertIn(untouched, cs.measurements)
        shim = LiveDataShim.__new__(LiveDataShim)
        shim.mirror_x = True
        result = measurement_from_dict(shim._twig_to_measurements(twig)[0])
        S = np.diag([-1, 1, 1])
        self.assertTrue(np.allclose(result.direction, S @ d))
        self.assertTrue(np.allclose(result.basis @ result.angular_covariance @ result.basis.T, S @ covariance @ S))


if __name__ == '__main__':
    unittest.main()
