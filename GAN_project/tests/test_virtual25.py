import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pipeline.fhcal_virtual25_gnn import (ModuleGeometry, PhysicalGeometry, VirtualCNNGenerator,
                                         GraphCritic, gradient_penalty, critic_loss, generator_loss)
from pipeline.fhcal_data import npz_grid_to_real_modules, ConditionEncoder, load_events
from pipeline.fhcal_training import train


torch.set_num_threads(1)


class Virtual25Tests(unittest.TestCase):
    def setUp(self):
        # Unequal-size touching modules, deliberately shifted off the 2.5 cm grid.
        self.geometry = PhysicalGeometry([ModuleGeometry(19, 18.2, 1., 900., 20., 20., 7),
                                          ModuleGeometry(8, .7, .3, 900., 15., 15., 10)])

    def test_area_mask_and_edges(self):
        g = self.geometry
        density = torch.ones(2, 10, g.height, g.width, requires_grad=True)
        energy = g.aggregate(density)
        torch.testing.assert_close(energy[0, 0], torch.full((10,), 225.))
        torch.testing.assert_close(energy[0, 1, :7], torch.full((7,), 400.))
        self.assertEqual(energy[:, 1, 7:].count_nonzero(), 0)
        self.assertEqual(g.edge_index.shape, (2, 2))
        self.assertAlmostEqual(g.edge_features[0, 3].item(), .75)
        energy.sum().backward()
        self.assertGreater(density.grad.abs().sum().item(), 0)
        self.assertTrue(((g.overlap > 0) & (g.overlap < 6.25)).any())

    def test_backprop_and_gp_mask(self):
        g = self.geometry
        generator, critic = VirtualCNNGenerator(g, 7, hidden=8), GraphCritic(g, 7, hidden=8)
        c = torch.randn(2, 7)
        fake = generator(torch.randn(2, 64), c)
        real = torch.rand_like(fake)*g.layer_mask
        loss = critic_loss(critic, real, fake, c)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        generator_loss(critic, generator(torch.randn(2, 64), c), c).backward()
        self.assertGreater(generator.seed.weight.grad.abs().sum().item(), 0)
        interpolated = real.requires_grad_(True)
        grad = torch.autograd.grad(critic(interpolated, c).sum(), interpolated)[0]
        self.assertEqual(grad[:, 1, 7:].count_nonzero(), 0)
        self.assertEqual(critic(real, c).shape, (2,))

    def test_mapping_and_conditions(self):
        g = self.geometry
        grid = np.zeros((2, 7, 9, 10), dtype=np.float32)
        grid[:, 0, 0] = 1; grid[:, 3, 4] = 2
        result = npz_grid_to_real_modules(grid, g, {'8': [0, 0], '19': [3, 4]})
        np.testing.assert_allclose(result.sum((1, 2)), grid.sum((1, 2, 3)))
        with self.assertRaises(ValueError):
            npz_grid_to_real_modules(grid, g, {'8': [0, 0], '19': [0, 0]})
        grid[:, 6, 8] = 3
        with self.assertRaises(ValueError):
            npz_grid_to_real_modules(grid, g, {'8': [0, 0], '19': [3, 4]})
        encoder = ConditionEncoder(torch.ones(2, 6), torch.tensor([13, 211]))
        restored = ConditionEncoder.from_state_dict(encoder.state_dict())
        self.assertTrue(torch.isfinite(restored.transform(torch.ones(2, 6), torch.tensor([13, 211]))).all())
        with self.assertRaises(ValueError):
            restored.transform(torch.ones(1, 6), torch.tensor([22]))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            PhysicalGeometry([ModuleGeometry(1, 0, 0, 0, 15, 15, 10), ModuleGeometry(2, 1, 0, 0, 20, 20, 7)])

    def test_training_checkpoint_resume(self):
        # Synthetic geometry, NOT a claim about actual FHCal centers.
        modules = [ModuleGeometry(i+1, i*22., 0., 900., 15. if i < 34 else 20.,
                                  15. if i < 34 else 20., 10 if i < 34 else 7) for i in range(54)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'geometry.json').write_text(json.dumps({'modules': [asdict(m) for m in modules]}))
            g = PhysicalGeometry(modules)
            energy = np.ones((6, 54, 10), dtype=np.float32)*g.layer_mask.numpy()
            np.savez(path/'events.npz', EnergyDeposit=energy, ModuleIds=np.arange(1, 55),
                     ParticlePoint=np.zeros((6, 3)), ParticleMomentum=np.ones((6, 3)), ParticlePDG=np.full(6, 13))
            args = SimpleNamespace(geometry=str(path/'geometry.json'), data=str(path/'events.npz'), module_map=None,
                                   output=str(path/'run'), epochs=1, batch_size=2, n_critic=1,
                                   seed=42, val_ratio=.33, device='cpu', resume=False)
            train(args)
            args.resume = True; args.epochs = 2
            train(args)
            saved = torch.load(path/'run/checkpoint.pt', weights_only=False)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(len((path/'run/metrics.jsonl').read_text().splitlines()), 2)
            # An uninterrupted run must reproduce resumed parameters.
            args.output = str(path/'continuous'); args.resume = False
            train(args)
            continuous = torch.load(path/'continuous/checkpoint.pt', weights_only=False)
            for name, parameter in saved['generator'].items():
                torch.testing.assert_close(parameter, continuous['generator'][name], rtol=0, atol=0)
            energy[:, 0, 9] = 0  # still valid
            energy[:, 53, 9] = 1  # absent layer must fail loudly
            np.savez(path/'invalid.npz', EnergyDeposit=energy, ModuleIds=np.arange(1, 55),
                     ParticlePoint=np.zeros((6, 3)), ParticleMomentum=np.ones((6, 3)), ParticlePDG=np.full(6, 13))
            with self.assertRaises(ValueError):
                load_events(path/'invalid.npz', g)


if __name__ == '__main__':
    unittest.main()
