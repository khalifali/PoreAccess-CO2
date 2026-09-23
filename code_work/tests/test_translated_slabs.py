"""Analytical checks for slab isolation and boundary handling."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_co2_translated_slabs import slab_conductance, reconstruct_inlet_labels
from extract_co2_pore_network import circumcenter_tetra


class SlabTests(unittest.TestCase):
    def test_inlet_rule_uses_face_beads_not_pore_height(self):
        particles = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,3.]])
        radii = np.full(4, .1)
        center, _ = circumcenter_tetra(particles)
        self.assertGreater(center[2], 1.25*2*radii[0])
        self.assertTrue(reconstruct_inlet_labels(center[None, :], particles, radii)[0])
        shift = np.array([0.,0.,10.])
        self.assertFalse(reconstruct_inlet_labels((center+shift)[None, :], particles+shift, radii)[0])
        self.assertTrue(reconstruct_inlet_labels((center+shift)[None, :], particles+shift, radii, bottom=10.)[0])

    def test_series_translation_and_scaling(self):
        xyz = np.column_stack([np.zeros(6), np.zeros(6), np.arange(6.)])
        edges = np.array([[i, i+1] for i in range(5)])
        for offset in [0, 9]:
            moved = xyz.copy(); moved[:, 2] += offset
            result = slab_conductance(moved, edges, np.ones(5)*10, offset, offset+5, .4)
            self.assertAlmostEqual(result['conductance_m3_s'], 2.)
            self.assertTrue(result['spanning'])
        self.assertAlmostEqual(slab_conductance(xyz, edges, np.ones(5)*20, 0, 5, .4)['conductance_m3_s'], 4.)

    def test_exterior_bypass_removed(self):
        xyz = np.array([[0,0,0], [0,0,2], [0,0,5], [0,0,-1.]])
        # Internal series conductance = 1. Exterior high-conductance bypass is excluded.
        edges = np.array([[0,1], [1,2], [0,3], [3,2]])
        result = slab_conductance(xyz, edges, np.array([2.,2.,100.,100.]), 0, 5, .4)
        self.assertAlmostEqual(result['conductance_m3_s'], 1.)
        self.assertEqual(result['throat_count'], 2)

    def test_disconnected_and_invalid_bands(self):
        xyz = np.array([[0,0,0], [0,0,2], [0,0,5.]])
        edges = np.array([[0,1]])
        result = slab_conductance(xyz, edges, np.ones(1), 0, 5, .4)
        self.assertEqual(result['conductance_m3_s'], 0.)
        self.assertFalse(result['spanning'])
        with self.assertRaises(ValueError):
            slab_conductance(xyz, edges, np.ones(1), 0, 5, 2.5)
        with self.assertRaises(ValueError):
            slab_conductance(xyz, edges, np.ones(1), .5, 5, .1)


if __name__ == '__main__':
    unittest.main()
