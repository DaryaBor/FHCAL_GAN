"""One combined NPZ trains the same one conditional model across 8/16/19."""
from pathlib import Path
import numpy as np
import pytest
from flow_pipeline.core import MODULES, KINEMATICS, load_three, beam_labels


def test_one_file_stratified_and_repeatable(tmp_path):
    rng=np.random.default_rng(42)
    energies=[];points=[];momentum=[];pdg=[]
    for module in MODULES:
        p,theta,phi=KINEMATICS[module]
        th,ph=np.deg2rad([theta,phi])
        vec=p*np.asarray([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
        n=35
        energy=np.zeros((n,7,5,7),dtype=np.float32)
        energy[:,1,0,1]=rng.uniform(.01,2.,n)
        energies.append(energy)
        points.append(rng.normal(size=(n,3)).astype(np.float32))
        momentum.append(np.tile(vec,(n,1)).astype(np.float32))
        pdg.append(np.full(n,-13,dtype=np.int32))
    perm=rng.permutation(105)
    path=tmp_path/'combined.npz'
    np.savez_compressed(path,EnergyDeposit=np.concatenate(energies)[perm],
                        ParticlePoint=np.concatenate(points)[perm],
                        ParticleMomentum=np.concatenate(momentum)[perm],
                        ParticlePDG=np.concatenate(pdg)[perm])
    ds=load_three(path,max_per_module=30,seed=11)
    assert tuple(ds.x.shape)==(90,7,7,5)
    assert tuple(ds.conditions.shape)==(90,5)
    assert len(ds.train_idx)==72 and len(ds.val_idx)==9 and len(ds.test_idx)==9
    assert [m['module'] for m in ds.metadata]==list(MODULES)
    assert [m['selected_events'] for m in ds.metadata]==[30,30,30]
    for mi in range(3):
        assert np.count_nonzero(ds.module_ids.numpy()[ds.train_idx]==mi)==24
    again=load_three(path,max_per_module=30,seed=11)
    assert np.array_equal(ds.x.numpy(),again.x.numpy())
    assert np.array_equal(ds.train_idx,again.train_idx)
    assert np.array_equal(ds.val_idx,again.val_idx)
    assert np.array_equal(ds.test_idx,again.test_idx)


def test_unknown_beam_must_fail():
    with pytest.raises(ValueError,match='Cannot identify incident module'):
        beam_labels(np.array([[0.,1.,0.]],dtype=np.float32))
