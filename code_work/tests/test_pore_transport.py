"""Analytical transport checks on tiny networks, independent of campaign outputs.
Run: python3 -m unittest discover -s tests -v (from code_work).
"""
import csv
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_co2_inlet_accessibility import effective_conductance
from extract_co2_pore_network import circumcenter_tetra, circumcenter_triangle
from solve_co2_pore_adsorption import graph_laplacian, incidence_weights, toth_equilibrium, toth_derivative


class TransportTests(unittest.TestCase):
    def test_series_parallel_and_disconnected(self):
        # Two series branches: 2,2 and 4,4 -> total conductance 1+2=3.
        edges=np.array([[0,1],[1,3],[0,2],[2,3]])
        source=np.array([1,0,0,0,0],bool); sink=np.array([0,0,0,1,0],bool)
        g, n, spans=effective_conductance(5,edges,np.array([2.,2.,4.,4.]),source,sink)
        self.assertAlmostEqual(g,3.); self.assertEqual(n,4); self.assertTrue(spans)
        source=np.array([0,0,0,0,1],bool)
        self.assertEqual(effective_conductance(5,edges,np.ones(4),source,sink)[0],0.)

    def test_laplacian_and_incidence(self):
        L=graph_laplacian(3,np.array([[0,1],[1,2]]),np.array([2.,3.]))
        np.testing.assert_allclose(L@np.ones(3),0.)
        self.assertAlmostEqual(float(np.sum(L@np.array([1.,3.,7.]))),0.)
        W=incidence_weights(2,3,np.array([0,0,1,1]),np.array([0,1,1,2]),np.array([1.,2.,3.]))
        np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(),1.)
        with self.assertRaises(ValueError):
            incidence_weights(3,3,np.array([0,1]),np.array([0,1]),np.ones(3))

    def test_isotherm_derivative(self):
        for exponent in [0.7,1.,1.3]:
            c=np.array([.1,1.,6.]); h=1e-6
            finite=(toth_equilibrium(c+h,298.15,5.332,5.093e-5,exponent)-
                    toth_equilibrium(c-h,298.15,5.332,5.093e-5,exponent))/(2*h)
            np.testing.assert_allclose(finite,toth_derivative(c,298.15,5.332,5.093e-5,exponent),rtol=1e-7)

    def test_circumcentres(self):
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        center,radius=circumcenter_tetra(xyz)
        np.testing.assert_allclose(center,[.5,.5,.5])
        center,radius=circumcenter_triangle(xyz[:3])
        np.testing.assert_allclose(center,[.5,.5,0.])

    def run_tiny_solver(self, boundary, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); net=root/'net'; net.mkdir(); out=root/'result'
            np.savez(net/'pore_network.npz',pore_xyz=[[0.,0.,.5],[0.,0.,1.5]],
                     pore_volume=[1.,1.],pore_inlet=[True,False],throat_conns=[[0,1]],
                     throat_area=[1.],throat_length=[1.],particle_id=[1],particle_radius=[.5])
            (net/'particle_pore_incidence.csv').write_text('pore_id,particle_index,particle_id\n0,0,1\n1,0,1\n')
            argv=['solver','--network',str(net),'--output',str(out),'--mode',mode,
                  '--diffusivity','1','--inlet-concentration','1','--inlet-boundary',boundary,
                  '--tube-radius','1','--particle-density','1','--qsat','1','--b-pa','.001',
                  '--k-ldf','.2','--t-end','2','--outputs','21','--rtol','1e-9','--atol','1e-12']
            def checked_solve(fun, interval, y0, **kwargs):
                if mode=='adsorption':
                    nf=2 if boundary=='finite' else 1
                    # Test both uptake and desorption branches away from switches.
                    for q in [.05,.95]:
                        y=np.r_[np.full(nf,.4),q,0.]
                        actual=kwargs['jac'](0.,y).toarray()
                        h=1e-6; I=np.eye(len(y))
                        numerical=np.column_stack([(fun(0.,y+h*v)-fun(0.,y-h*v))/(2*h) for v in I])
                        np.testing.assert_allclose(actual,numerical,atol=1e-8,rtol=1e-6)
                        dy=fun(0.,y)
                        self.assertAlmostEqual(dy[:nf].sum()+np.pi/6*dy[nf]-dy[-1],0.,places=12)
                return scipy_solve(fun,interval,y0,**kwargs)
            with patch.object(sys,'argv',argv), patch('scipy.integrate.solve_ivp',checked_solve):
                runpy.run_path(str(Path(__file__).resolve().parents[1]/'solve_co2_pore_adsorption.py'),run_name='__main__')
            with (out/'timeseries.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertLess(max(abs(float(x['mass_balance_residual_mol'])) for x in rows),1e-8)
            state=np.load(out/'final_state.npz')
            self.assertGreaterEqual(state['pore_concentration'].min(),-1e-10)
            if mode=='diffusion' and boundary=='fixed':
                # Unit pore volume/conductance, c(0)=0, fixed neighbouring c=1.
                self.assertAlmostEqual(state['pore_concentration'][1],1-np.exp(-2),places=7)

    def test_fixed_diffusion_analytic(self): self.run_tiny_solver('fixed','diffusion')
    def test_finite_diffusion_balance(self): self.run_tiny_solver('finite','diffusion')
    def test_fixed_adsorption_jacobian_balance(self): self.run_tiny_solver('fixed','adsorption')
    def test_finite_adsorption_jacobian_balance(self): self.run_tiny_solver('finite','adsorption')

if __name__=='__main__': unittest.main()
