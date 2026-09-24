"""Invariant and gradient checks for the residual regional GNN."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from compare_co2_gnn_v2 import RegionalGNN,graph_v2,pack_v2,KNOTS,QEQ


class RegionalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(11)
        self.xyz=np.array([[0.,0.,.001],[.0016,0.,.001],[0.,.0016,.004],[0.,0.,.0056]])
        self.model=RegionalGNN(.3*(1-np.exp(-KNOTS/2000)))

    def test_permutation_and_rotation(self):
        batch=pack_v2([graph_v2(self.xyz)])
        with torch.inference_mode():expected=self.model(batch).numpy()
        rotated=self.xyz.copy();rotated[:,0]=-self.xyz[:,1];rotated[:,1]=self.xyz[:,0]
        for points in [self.xyz[[2,0,3,1]],rotated]:
            with torch.inference_mode():got=self.model(pack_v2([graph_v2(points)])).numpy()
            np.testing.assert_allclose(got,expected,rtol=1e-5,atol=1e-7)

    def test_batch_independence_bounds_and_gradients(self):
        other=self.xyz.copy();other[:,0]+=.0001
        with torch.inference_mode():
            one=self.model(pack_v2([graph_v2(self.xyz)]))
            two=self.model(pack_v2([graph_v2(self.xyz),graph_v2(other)]))
        np.testing.assert_allclose(one.numpy()[0],two.numpy()[0],rtol=1e-5,atol=1e-7)
        self.assertTrue(torch.all(two[:,0]==0))
        self.assertTrue(torch.all(torch.diff(two,dim=1)>=0))
        self.assertTrue(torch.all(two<=QEQ+1e-7))
        pred=self.model(pack_v2([graph_v2(self.xyz)]));pred[:,-1].sum().backward()
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__=='__main__':unittest.main()
