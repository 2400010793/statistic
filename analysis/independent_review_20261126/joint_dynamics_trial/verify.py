"""Target-free checks for the coupled loss, integer allocation and ODE discretization."""
import joblib
from threadpoolctl import threadpool_limits
from core import OUT,SEEDS,np,torch,Dynamics,distribution_loss,allocation,save

def main():
    torch.set_num_threads(2);torch.manual_seed(15)
    net=Dynamics(True);x=torch.randn(128,32);y=torch.randn(128,32)+.2
    z,w,_,_,_=net(x);loss,_=distribution_loss(z,y,w,64.)
    loss.backward();grad=net.head.weight.grad
    grads=dict(velocity=float(torch.linalg.vector_norm(grad[:32])),growth=float(torch.linalg.vector_norm(grad[32])))
    assert min(grads.values())>0,'Both heads must receive gradients from the same endpoint objective'
    np.testing.assert_array_equal(allocation(np.full(2000,1/2000),0),np.arange(2000))
    space=joblib.load(OUT/'representation.joblib');zz=np.load(OUT/'early_latent.npz')['z9'][:1000]/space['scale']
    arr=torch.from_numpy(zz.astype(np.float32));reports=[]
    for seed in SEEDS:
        for kind in ['velocity','joint']:
            ck=torch.load(OUT/f'{kind}_{seed}.pt',weights_only=True);net=Dynamics(ck['growth']);net.load_state_dict(ck['state'])
            with torch.no_grad():
                z4,w4,_,_,_=net(arr,steps=4);z16,w16,_,_,_=net(arr,steps=16)
            reports.append(dict(seed=seed,kind=kind,relative_displacement_change_4_to_16=float(torch.linalg.vector_norm(z16-z4)/torch.linalg.vector_norm(z4-arr)),
                weight_l1_4_to_16=float(torch.sum(abs(w16-w4)))))
    save('verification.json',dict(coupled_loss_head_gradient_norms=grads,uniform_allocation_exact_identity=True,
        ode_sensitivity=reports,note='Sensitivity only; frozen forecasts remain 4 steps. No target data accessed.'))
    print(grads,reports)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
