"""Check feedback gradients, normalization, cumulative mass and budget semantics."""
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def main():
    torch.set_num_threads(2);rep=joblib.load(JOINT/'representation.joblib')
    decoder=Decoder(rep['pca'],rep['scale']);old=v1.Decoder(rep['pca'],rep['scale'])
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    src=np.load(ROOT/'analysis/results/p3_preparation/damped_group_forecast/source_rows_E10.5.npy')
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);base=torch.from_numpy(x[src[:12]].toarray());del x
    torch.manual_seed(39);net=StageDynamics(True)
    with torch.no_grad():
        net.base.weight.normal_(0,.02);net.residual.weight.normal_(0,.01)
    a,b=pair(net,decoder,base,0)
    b_alone=day(net,decoder,a['expression'],1,anchor=base)
    torch.testing.assert_close(b['logw'],a['logw']+b_alone['logw'],rtol=0,atol=0)
    q=torch.full((len(base),),1/len(base));w,_,ess=weights_and_penalty(b['logw'],q)
    torch.testing.assert_close(ess,1/(len(w)*w.square().sum()))
    ga=torch.autograd.grad(b['z'].square().mean(),a['expression'],retain_graph=True)[0]
    assert torch.isfinite(ga).all() and ga.abs().max()>0
    for r in [a,b]:
        assert torch.isfinite(r['expression']).all() and (r['expression']>=0).all()
        torch.testing.assert_close(torch.expm1(r['expression']).sum(1),torch.full((len(base),),10000.),rtol=1e-5,atol=.01)
        assert torch.equal(r['expression']>0,base>0)
    z=decoder.project(base);delta=torch.randn_like(z)*.3
    torch.testing.assert_close(decoder(base,delta),old(base,delta),rtol=0,atol=0)
    # Strong edits activate the total budget and must match an independent formula.
    middle=base*2;raw_delta=torch.full_like(z,20.)
    got,diag=decoder(middle,raw_delta,anchor=base,diagnostics=True)
    shift=raw_delta*rep['scale']@decoder.axes
    edited=middle*torch.clamp(1+shift/(middle+.25),.8,1.2)
    capped=torch.clamp(edited,min=base*.64,max=base*1.44)
    linear=torch.expm1(capped);expected=torch.log1p(linear*10000/linear.sum(1,keepdim=True))
    torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-6);assert diag['total_bound_count']>0
    # Zero vector field yields identity distribution; two normalizations only add rounding.
    zero=StageDynamics(True);_,identity=pair(zero,decoder,base,2)
    torch.testing.assert_close(identity['expression'],base,rtol=2e-6,atol=2e-6)
    # Check actual two-day objective gradient with a centered difference in double precision.
    net=net.double();decoder=decoder.double();bb=base.double()
    for p in net.parameters():p.requires_grad_(False)
    bias=net.base.bias;bias.requires_grad_(True)
    _,out=pair(net,decoder,bb,0)
    val=out['z'].square().mean()+out['logw'].square().mean()
    grad=torch.autograd.grad(val,bias)[0];direction=torch.randn_like(bias);direction/=direction.norm()
    analytic=float(grad@direction);eps=1e-5;original=bias.detach().clone();values=[]
    for sign in [1,-1]:
        with torch.no_grad():bias.copy_(original+sign*eps*direction)
        _,out=pair(net,decoder,bb,0);values.append(float((out['z'].square().mean()+out['logw'].square().mean()).detach()))
    fd=(values[0]-values[1])/(2*eps)
    np.testing.assert_allclose(analytic,fd,rtol=.002,atol=2e-6)
    save('verification.json',dict(passed=True,original_decoder_exact_match=True,cumulative_logmass_exact=True,
        second_day_backpropagates_to_first_expression=True,zero_field_identity=True,
        total_edit_budget_check=True,nonnegative_finite_cp10k_support_checks=True,
        directional_gradient=analytic,finite_difference=fd))
    print('Preflight passed',analytic,fd,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
