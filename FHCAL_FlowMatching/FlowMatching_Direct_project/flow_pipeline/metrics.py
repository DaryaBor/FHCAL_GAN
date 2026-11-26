"""Common central-only Geant4 vs generated-energy metrics for all experiments."""
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

def metric(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    res=ks_2samp(a,b,method='auto')
    maximum=max(float(np.quantile(np.concatenate([a,b]),0.9995)),1e-6)
    edges=np.linspace(0,maximum,81)
    h1=np.histogram(np.clip(a,0,maximum-1e-12),bins=edges)[0].astype(float)+1e-12
    h2=np.histogram(np.clip(b,0,maximum-1e-12),bins=edges)[0].astype(float)+1e-12
    js=float(jensenshannon(h1/h1.sum(),h2/h2.sum(),base=2.0)**2)
    return {'ks':float(res.statistic),'pvalue':float(res.pvalue),'js_bits':js,
            'real_mean':float(a.mean()),'fake_mean':float(b.mean()),
            'real_std':float(a.std()),'fake_std':float(b.std()),
            'real_q99':float(np.quantile(a,0.99)), 'fake_q99':float(np.quantile(b,0.99))}


def draw(true,fake,folder,title):
    folder.mkdir(parents=True,exist_ok=True)
    te=true.sum(axis=(1,2,3));fe=fake.sum(axis=(1,2,3))
    m=metric(te,fe)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(te,bins=70,histtype='step',density=True,label='Geant4',linewidth=1.6)
    ax.hist(fe,bins=70,histtype='step',density=True,label='Flow Matching',linewidth=1.6)
    ax.set(title=title+' — total energy',xlabel='Total E, dataset units',ylabel='density')
    ax.text(.02,.98,f"KS={m['ks']:.4f}, p={m['pvalue']:.2e}, JS={m['js_bits']:.4f} bits",transform=ax.transAxes,va='top')
    ax.legend();fig.tight_layout();fig.savefig(folder/'total_energy.png',dpi=150);plt.close(fig)
    layer_r=true.sum(axis=(2,3));layer_f=fake.sum(axis=(2,3))
    fig,axs=plt.subplots(2,4,figsize=(17,7));axs=axs.flatten()
    for i in range(7):
        lm=metric(layer_r[:,i],layer_f[:,i]);ax=axs[i]
        ax.hist(layer_r[:,i],bins=35,density=True,histtype='step',label='Geant4')
        ax.hist(layer_f[:,i],bins=35,density=True,histtype='step',label='FM')
        ax.set_title(f'L{i+1}: KS {lm["ks"]:.3f}, p {lm["pvalue"]:.1e}',fontsize=10)
    axs[-1].axis('off');axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'layer_distributions.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,8),layer_r.mean(axis=0),'-o',label='Geant4')
    ax.plot(range(1,8),layer_f.mean(axis=0),'-o',label='FM');ax.legend();ax.set(title=title+' — longitudinal profile',xlabel='Layer',ylabel='Mean E per event');fig.tight_layout();fig.savefig(folder/'longitudinal_profile.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,8),(true>0).mean(axis=(0,2,3)),'-o',label='Geant4')
    ax.plot(range(1,8),(fake>0).mean(axis=(0,2,3)),'-o',label='FM');ax.legend();ax.set(title=title+' — occupancy E>0',xlabel='Layer',ylabel='Fraction active');fig.tight_layout();fig.savefig(folder/'occupancy.png',dpi=150);plt.close(fig)
    a=true.sum(axis=1).mean(axis=0);b=fake.sum(axis=1).mean(axis=0)
    vmax=max(a.max(),b.max())
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].imshow(a,origin='lower',vmin=0,vmax=vmax);im=ax[1].imshow(b,origin='lower',vmin=0,vmax=vmax)
    ax[0].set_title('Geant4');ax[1].set_title('Flow Matching');fig.colorbar(im,ax=ax.tolist(),shrink=.8);fig.savefig(folder/'spatial_map.png',dpi=150);plt.close(fig)
    return m


def save_report(real, fake, labels, out, meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {'metadata':meta, 'per_condition':{}}
    summaries, layers = [], []
    for module in (8,16,19,'ALL'):
        selection = labels == module if module != 'ALL' else np.ones(len(labels), dtype=bool)
        rr, ff = real[selection], fake[selection]
        if len(rr) == 0: continue
        m = draw(rr, ff, out/str(module), f'FHCAL: {module}')
        m.update({'module':module,'n_real':len(rr),'n_fake':len(ff),
                  'occupancy_real':float((rr>0).mean()),'occupancy_fake':float((ff>0).mean())})
        results['per_condition'][str(module)] = m
        summaries.append(m)
        for l in range(7):
            v=metric(rr[:,l].sum((1,2)),ff[:,l].sum((1,2)))
            layers.append({'module':module,'layer':l+1,**v,
                           'occupancy_real':float((rr[:,l]>0).mean()),'occupancy_fake':float((ff[:,l]>0).mean())})
    (out/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    for name,rows in [('summary_metrics.csv',summaries),('layer_metrics.csv',layers)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return results
