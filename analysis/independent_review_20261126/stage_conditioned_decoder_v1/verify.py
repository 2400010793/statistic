"""Numerical and provenance checks relevant to the new differentiable model."""
import argparse
import importlib.util
import json
import joblib
from threadpoolctl import threadpool_limits
from model import OUT,REVIEW,JOINT,ROOT,prior,np,torch,SEEDS,ARMS,StageDynamics,Decoder,weights_and_penalty,stratified_batch,save

def main(after_training=False):
    torch.set_num_threads(2);torch.manual_seed(20261128)
    d=np.load(OUT/'data_splits.npz');rep=joblib.load(JOINT/'representation.joblib')
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines();x=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    base=x[d['sources'][0][:16]].toarray();decoder=Decoder(rep['pca'],rep['scale'])
    delta=torch.randn(16,32)*.03
    with torch.no_grad():got=decoder(torch.from_numpy(base),delta).numpy()
    spec=importlib.util.spec_from_file_location('old_joint_core',JOINT/'core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=module.decode(base,(delta.numpy()*rep['scale'])@rep['pca'].components_)
    np.testing.assert_allclose(got,expected,rtol=2e-5,atol=3e-6)
    np.testing.assert_array_equal(got>0,base>0)
    np.testing.assert_allclose(np.expm1(got.astype(float)).sum(1),10000,rtol=1e-5)
    exact=Decoder(rep['pca'],rep['scale']).double();xx=torch.from_numpy(base).double();dz=delta.double().requires_grad_(True)
    direction=torch.randn_like(dz);direction=direction/direction.norm();eps=1e-5
    loss=exact.project(exact(xx,dz)).square().mean();loss.backward();analytic=float((dz.grad*direction).sum())
    with torch.no_grad():
        plus=exact.project(exact(xx,dz+eps*direction)).square().mean();minus=exact.project(exact(xx,dz-eps*direction)).square().mean()
    numerical=float((plus-minus)/(2*eps));assert abs(analytic-numerical)<1e-5+abs(analytic)*1e-3
    rng=np.random.default_rng(25);ix,q=stratified_batch(rng,d['train9'],d['groups9']);qt=torch.from_numpy(q)
    w,pen,ess=weights_and_penalty(torch.zeros(len(q)),qt)
    np.testing.assert_allclose(w.numpy(),q,rtol=1e-6,atol=1e-8);assert abs(float(ess)-1)<1e-6 and float(pen)==0
    group_weights=np.bincount(d['groups9'][ix],weights=w.numpy(),minlength=12)
    population=np.bincount(d['groups9'][d['train9']],minlength=12)/len(d['train9'])
    np.testing.assert_allclose(group_weights,population,atol=1e-7)
    concentrated=torch.zeros(len(q),requires_grad=True);offset=torch.zeros(len(q));offset[0]=7
    _,penalty,_=weights_and_penalty(concentrated+offset,qt);penalty.backward()
    assert penalty>0 and concentrated.grad[0]>0
    model=StageDynamics(True)
    with torch.no_grad():model.base.weight.normal_(std=.01);model.residual.weight.normal_(std=.01)
    z=torch.randn(16,32)
    with torch.no_grad():
        v0,g0,r=model.field(z,.25);v1,g1,_=model.field(z,1.25)
    np.testing.assert_allclose((v1-v0).numpy(),r[:,:32].numpy(),rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose((g1-g0).numpy(),r[:,32].numpy(),rtol=1e-5,atol=1e-7)
    report=dict(decoder_max_absolute_difference=float(np.max(abs(got-expected))),decoder_directional_derivative=dict(analytic=analytic,finite_difference=numerical),
        zero_support_and_cp10k_passed=True,source_strata_exact_probability_correction=True,ess_penalty_has_concentration_gradient=True,
        stage_residual_field_identity_passed=True)
    if after_training:
        training=json.loads((OUT/'training.json').read_text());assert prior.digest(OUT/'frozen_predictions.npz')==training['frozen_sha256']
        frozen=np.load(OUT/'frozen_predictions.npz');replays=[];time_effects=[]
        for fold in range(2):
            for seed in SEEDS:
                for arm in ARMS:
                    key=f'f{fold}_s{seed}_{arm}';ck=torch.load(OUT/f'{key}.pt',weights_only=True)
                    model=StageDynamics(arm!='autonomous_latent');model.load_state_dict(ck['state']);model.eval()
                    for phase,pool,src in [(0,0,d['test8']),(1,1,d['sources'][0])]:
                        zz=d['z8'][src] if phase==0 else d['z9'][src];zt=torch.from_numpy((zz/rep['scale']).astype(np.float32))
                        with torch.no_grad():
                            out,lw,_,_,_=model(zt,phase);fine,flw,_,_,_=model(zt,phase,steps=32)
                        error=float(np.max(abs(out.numpy()*rep['scale']-frozen[key+'_future'][pool])))
                        np.testing.assert_allclose(out.numpy()*rep['scale'],frozen[key+'_future'][pool],rtol=1e-6,atol=1e-5)
                        np.testing.assert_allclose(lw.numpy(),frozen[key+'_logw'][pool],rtol=1e-6,atol=1e-6)
                        relative=float(torch.linalg.vector_norm(fine-out)/max(float(torch.linalg.vector_norm(out-zt)),1e-12))
                        replays.append(dict(fold=fold,seed=seed,arm=arm,phase=phase,replay_max_error=error,integration_relative_change=relative))
                        if phase==1:
                            with torch.no_grad():
                                va,ga,_=model.field(zt,.5);vb,gb,_=model.field(zt,1.5)
                            cosine=(va*vb).sum(1)/(va.norm(dim=1)*vb.norm(dim=1)).clamp_min(1e-12)
                            time_effects.append(dict(fold=fold,seed=seed,arm=arm,velocity_change_rms=float((va-vb).square().mean().sqrt()),
                                velocity_cosine_median=float(cosine.median()),growth_change_rms=float((ga-gb).square().mean().sqrt()),
                                interpretation='Change model time while holding E9.5 expression fixed; not measured biological velocity'))
        report['checkpoint_replays']=replays;report['time_condition_effects']=time_effects
        report['frozen_unchanged']=prior.digest(OUT/'frozen_predictions.npz')==training['frozen_sha256']
    save('verification.json' if after_training else 'preflight_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoint_replays'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--after-training',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.after_training)
