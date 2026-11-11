"""Physical-space WGAN-GP training, validation, logging and exact resume."""
import json
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .fhcal_data import ConditionEncoder, load_events, load_geometry
from .fhcal_virtual25_gnn import VirtualCNNGenerator, GraphCritic, critic_loss, generator_loss


def distribution_metrics(real, fake):
    """Empirical KS/W1 and histogram JS; empty positive tails are explicit."""
    if not len(real) or not len(fake):
        return {'ks': None, 'w1': None, 'js': None}
    real, fake = real.sort().values.contiguous(), fake.sort().values.contiguous()
    support = torch.cat((real, fake)).sort().values
    cr = torch.searchsorted(real, support, right=True).float()/len(real)
    cf = torch.searchsorted(fake, support, right=True).float()/len(fake)
    distance = (cr-cf).abs()
    lo, hi = support[0].item(), support[-1].item()
    if hi <= lo:
        js = 0.
    else:
        pr = torch.histc(real, bins=50, min=lo, max=hi)/len(real)
        pf = torch.histc(fake, bins=50, min=lo, max=hi)/len(fake)
        middle = ((pr+pf)/2).clamp_min(1e-30)
        js = (.5*(pr*(pr.clamp_min(1e-30)/middle).log()).sum()+
              .5*(pf*(pf.clamp_min(1e-30)/middle).log()).sum()).item()
    return {'ks': distance.max().item(),
            'w1': (distance[:-1]*(support[1:]-support[:-1])).sum().item(), 'js': js}


@torch.no_grad()
def physical_metrics(real, fake, geometry, threshold=1e-6):
    """Validation in physical energy units, including zero-signal mismatch."""
    r, f = real.sum(-1), fake.sum(-1)
    centers = torch.tensor([[m.x, m.y] for m in geometry.modules], device=real.device)
    def shower(e):
        total = e.sum(1).clamp_min(1e-12)
        center = (e[..., None]*centers).sum(1)/total[:, None]
        radius = ((e*(centers[None]-center[:, None]).square().sum(-1)).sum(1)/total).sqrt()
        return center, radius
    rc, rr = shower(r); fc, fr = shower(f)
    result = {'total_mean_real': r.sum(1).mean().item(), 'total_mean_fake': f.sum(1).mean().item(),
              'layer_mean_abs_error': (real.mean(0)-fake.mean(0)).abs().mean().item(),
              'centroid_mean_distance': (rc.mean(0)-fc.mean(0)).norm().item(),
              'radius_mean_real': rr.mean().item(), 'radius_mean_fake': fr.mean().item()}
    result.update({f'total_{k}': v for k, v in distribution_metrics(r.sum(1), f.sum(1)).items()})
    small = torch.tensor([m.width == 15 for m in geometry.modules], device=real.device)
    for label, e in (('real', r), ('fake', f)):
        result[f'inner_fraction_{label}'] = (e[:, small].sum(1)/e.sum(1).clamp_min(1e-12)).mean().item()
    for k, m in enumerate(geometry.modules):
        for label, e in (('real', r), ('fake', f)):
            result[f'module_{m.module_id}_{label}_zero'] = (e[:, k] == 0).float().mean().item()
            result[f'module_{m.module_id}_{label}_near_zero'] = (e[:, k] <= threshold).float().mean().item()
        for suffix, a, b in (('', r[:, k], f[:, k]), ('_positive', r[r[:, k] > 0, k], f[f[:, k] > 0, k])):
            result.update({f'module_{m.module_id}{suffix}_{name}': value
                           for name, value in distribution_metrics(a, b).items()})
    source, target = geometry.edge_index
    if len(source):
        def correlations(e):
            centered = e-e.mean(0)
            std = centered.square().mean(0).sqrt().clamp_min(1e-12)
            return (centered[:, source]*centered[:, target]).mean(0)/(std[source]*std[target])
        result['neighbor_correlation_mae'] = (correlations(r)-correlations(f)).abs().mean().item()
    return result


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.n_critic < 1 or not 0 < args.val_ratio < 1:
        raise ValueError('Invalid training parameters')
    torch.manual_seed(args.seed)
    geometry = load_geometry(args.geometry)
    energy, values, pdg = load_events(args.data, geometry, args.module_map)
    if len(energy) < 2:
        raise ValueError('At least two events needed')
    order = torch.randperm(len(energy))
    val_count = min(len(energy)-1, max(1, int(len(energy)*args.val_ratio)))
    val_idx, train_idx = order[:val_count], order[val_count:]
    checkpoint_path = Path(args.output)/'checkpoint.pt'
    saved = None
    if checkpoint_path.exists() and not args.resume:
        raise ValueError('Output already has a checkpoint: use --resume or a new --output')
    digest = hashlib.sha256()
    with open(args.data, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    data_hash = digest.hexdigest()
    if args.resume:
        # Load only trusted local checkpoints (includes optimizer/RNG state).
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        signature = {'geometry': [asdict(m) for m in geometry.modules], 'data': str(Path(args.data).resolve()),
                     'module_map': Path(args.module_map).read_text() if args.module_map else None,
                     'data_sha256': data_hash, 'n_events': len(energy), 'seed': args.seed, 'val_ratio': args.val_ratio,
                     'batch_size': args.batch_size, 'n_critic': args.n_critic}
        if saved['signature'] != signature:
            raise ValueError('Resume inputs/configuration differ from checkpoint')
        train_idx, val_idx = saved['train_idx'], saved['val_idx']
        encoder = ConditionEncoder.from_state_dict(saved['conditions'])
    else:
        encoder = ConditionEncoder(values[train_idx], pdg[train_idx])
    conditions = encoder.transform(values, pdg)
    device = torch.device(args.device)
    generator = VirtualCNNGenerator(geometry, conditions.size(1)).to(device)
    critic = GraphCritic(geometry, conditions.size(1)).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0., .9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0., .9))
    start, steps = 0, 0
    if saved:
        generator.load_state_dict(saved['generator']); critic.load_state_dict(saved['critic'])
        opt_g.load_state_dict(saved['opt_g']); opt_d.load_state_dict(saved['opt_d'])
        start, steps = saved['epoch'], saved['steps']
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    loader = DataLoader(TensorDataset(energy[train_idx], conditions[train_idx]), batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(TensorDataset(energy[val_idx], conditions[val_idx]), batch_size=args.batch_size)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    signature = {'geometry': [asdict(m) for m in geometry.modules], 'data': str(Path(args.data).resolve()),
                 'module_map': Path(args.module_map).read_text() if args.module_map else None,
                 'data_sha256': data_hash, 'n_events': len(energy), 'seed': args.seed, 'val_ratio': args.val_ratio,
                 'batch_size': args.batch_size, 'n_critic': args.n_critic}
    for epoch in range(start, args.epochs):
        generator.train(); critic.train()
        d_losses, g_losses = [], []
        for real, c in loader:
            real, c = real.to(device), c.to(device)
            with torch.no_grad():
                fake = generator(torch.randn(len(real), generator.noise_dim, device=device), c)
            opt_d.zero_grad(set_to_none=True)
            loss_d = critic_loss(critic, real, fake, c)
            loss_d.backward(); opt_d.step()
            d_losses.append(loss_d.item()); steps += 1
            if steps % args.n_critic == 0:
                critic.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                fake = generator(torch.randn(len(real), generator.noise_dim, device=device), c)
                loss_g = generator_loss(critic, fake, c)
                loss_g.backward(); opt_g.step()
                critic.requires_grad_(True)
                g_losses.append(loss_g.item())
        generator.eval(); critic.eval()
        # Validation has its own RNG stream so it does not perturb training noise.
        real_batches, fake_batches = [], []
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.manual_seed(args.seed+epoch)
            with torch.no_grad():
                for real, c in validation:
                    c = c.to(device)
                    real_batches.append(real.to(device))
                    fake_batches.append(generator(torch.randn(len(real), generator.noise_dim, device=device), c))
        val_real, val_fake = torch.cat(real_batches), torch.cat(fake_batches)
        report = physical_metrics(val_real, val_fake, geometry)
        # Same condition subsets for real and generated events; vertex != impact.
        groups = {f'pdg_{code}': pdg[val_idx] == code for code in encoder.pdg_codes}
        for axis, label in enumerate(('vertex_x', 'vertex_y')):
            median = values[train_idx, axis].median()
            groups[f'{label}_low'] = values[val_idx, axis] <= median
            groups[f'{label}_high'] = values[val_idx, axis] > median
        for label, mask in groups.items():
            mask = mask.to(device)
            if mask.any():
                report.update({f'{label}_total_{k}': v for k, v in distribution_metrics(
                    val_real[mask].sum((1, 2)), val_fake[mask].sum((1, 2))).items()})
        report.update(epoch=epoch+1, critic_loss=sum(d_losses)/len(d_losses),
                      generator_loss=sum(g_losses)/len(g_losses) if g_losses else None)
        with (Path(args.output)/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        print(json.dumps({k: report[k] for k in ('epoch', 'critic_loss', 'generator_loss', 'total_mean_real', 'total_mean_fake')}), flush=True)
        state = dict(epoch=epoch+1, steps=steps, generator=generator.state_dict(), critic=critic.state_dict(),
                     opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(), conditions=encoder.state_dict(),
                     signature=signature, train_idx=train_idx, val_idx=val_idx, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(checkpoint_path)
    return generator, critic, encoder
