"""Structural checks independent of campaign prediction scores."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare_co2_gnn_uptake import particle_graph,pack,CurveGNN,KNOTS,QEQ,interpolation_matrix,time_weights


class GNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.xyz=np.array([[0.,0.,.001],[.0016,0.,.001],[0.,.0016,.001],[0.,0.,.0026]])
        torch.manual_seed(7)
        self.model=CurveGNN(.3*(1-np.exp(-KNOTS/2000)))
        self.model.eval()

    def predict(self,positions):
        with torch.inference_mode():return self.model(pack([particle_graph(positions)])).numpy()

    def test_permutation_and_axial_rotation_invariance(self):
        expected=self.predict(self.xyz)
        np.testing.assert_allclose(expected,self.predict(self.xyz[[3,0,2,1]]),rtol=1e-6,atol=1e-8)
        rotated=self.xyz.copy();rotated[:,0]=-self.xyz[:,1];rotated[:,1]=self.xyz[:,0]
        np.testing.assert_allclose(expected,self.predict(rotated),rtol=1e-6,atol=1e-8)

    def test_batch_has_no_cross_bed_effect(self):
        one=particle_graph(self.xyz);two=particle_graph(self.xyz+np.array([0,0,.002]))
        with torch.inference_mode():batch=self.model(pack([one,two])).numpy()
        np.testing.assert_allclose(batch[0],self.predict(self.xyz)[0],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(batch[1],self.predict(self.xyz+np.array([0,0,.002]))[0],rtol=1e-6,atol=1e-8)

    def test_curve_bounds_and_interpolation(self):
        knots=self.predict(self.xyz)[0]
        times=np.linspace(0,5000,101)
        curve=interpolation_matrix(times)@knots
        self.assertEqual(curve[0],0.)
        self.assertTrue(np.all(np.diff(curve)>=0))
        self.assertLessEqual(curve[-1],QEQ)
        np.testing.assert_allclose(interpolation_matrix(KNOTS),np.eye(len(KNOTS)))
        self.assertAlmostEqual(time_weights(times).sum(),1.)


if __name__=='__main__':unittest.main()
