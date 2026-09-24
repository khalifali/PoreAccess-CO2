import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from compare_co2_gnn_diffusivity import DiffusivityGNN,graph_v2,pack_v2,DP
class DiffusivityTests(unittest.TestCase):
    def test_scalar_invariance_and_gradients(self):
        torch.manual_seed(1)
        xyz=np.array([[0,0,.5],[.2,0,1.5],[0,.1,3],[0,0,4]])*DP
        g=graph_v2(xyz);m=DiffusivityGNN(-12,.3)
        out=m(pack_v2([g]));other=m(pack_v2([graph_v2(xyz[::-1].copy())]))
        self.assertEqual(tuple(out.shape),(1,));torch.testing.assert_close(out,other)
        self.assertTrue(torch.exp(out).item()>0)
        batched=m(pack_v2([g,g]));torch.testing.assert_close(batched,out.expand(2))
        out.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
