"""Verify frozen replay, direction constraints, saturation and integrator sensitivity."""
import json
import joblib
from threadpoolctl import threadpool_limits
from helpers import OUT,JOINT,prior,np,torch,SEEDS,MODES,Calibrated,load_original,save

def main():
    torch.set_num_threads(2)
    fit=json.loads((OUT/'calibration.json').read_text());assert prior.digest(OUT/'frozen_predictions.npz')==fit['frozen_sha256']
    d=np.load(OUT/'data_splits.npz');f=np.load(OUT/'frozen_predictions.npz');old=np.load(JOINT/'frozen_predictions.npz')
    scale=joblib.load(JOINT/'representation.joblib')['scale'];src=d['sources'][0]
    x=torch.from_numpy((d['z9'][src]/scale).astype(np.float32));checks=[];replay=[]
    for seed in SEEDS:
        original=load_original(seed)
        with torch.no_grad():z,_,lw,_,_=original(x,steps=4)
        err=float(np.max(abs(z.numpy()*scale-old[f'joint_{seed}_h1_future'][0])))
        np.testing.assert_allclose(z.numpy()*scale,old[f'joint_{seed}_h1_future'][0],rtol=1e-6,atol=1e-5)
        np.testing.assert_allclose(lw.numpy(),old[f'joint_{seed}_h1_logw'][0],rtol=1e-6,atol=1e-6)
        replay.append(dict(seed=seed,max_absolute_old_prediction_error=err))
        for fold in range(2):
            for mode in MODES:
                key=f'f{fold}_s{seed}_{mode}';checkpoint=torch.load(OUT/f'{key}.pt',weights_only=True)
                net=Calibrated(original,mode);net.load_state_dict(checkpoint['state']);net.eval()
                with torch.no_grad():
                    z16,w16,l16,_,_=net(x,steps=16);z32,w32,l32,_,_=net(x,steps=32)
                    v,g,s=net.field(x);vo=original.head(original.trunk(x))[:,:32]
                np.testing.assert_allclose(z16.numpy()*scale,f[key+'_future'][0],rtol=1e-6,atol=1e-5)
                np.testing.assert_allclose(l16.numpy(),f[key+'_logw'][0],rtol=1e-6,atol=1e-6)
                error=float(torch.linalg.vector_norm(v-s[:,None]*vo)/max(float(torch.linalg.vector_norm(v)),1e-12))
                if mode!='full':assert error<1e-6
                checks.append(dict(fold=fold,seed=seed,mode=mode,direction_constraint_relative_error=error if mode!='full' else None,
                    speed_min=float(s.min()),speed_max=float(s.max()),speed_near_upper_fraction=float((s>=7.6).float().mean()),
                    instantaneous_growth_near_bound_fraction=float((abs(g)>=.95*np.log(4)).float().mean()),
                    endpoint_logweight_near_bound_fraction=float((abs(l16)>=.95*np.log(4)).float().mean()),
                    relative_displacement_16_to_32=float(torch.linalg.vector_norm(z32-z16)/max(float(torch.linalg.vector_norm(z16-x)),1e-12)),
                    weights_l1_16_to_32=float(abs(w16-w32).sum())))
    save('verification.json',dict(old_replay=replay,checks=checks,frozen_unchanged=prior.digest(OUT/'frozen_predictions.npz')==fit['frozen_sha256']))
    print('Verified',len(checks),'models; old replay errors',[r['max_absolute_old_prediction_error'] for r in replay])

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
