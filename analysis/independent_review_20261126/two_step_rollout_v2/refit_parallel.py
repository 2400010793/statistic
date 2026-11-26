"""Equivalent full-P3 refits, two independent processes with two CPU threads each."""
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from train import fit,load_data,threadpool_limits
from rollout import *

def worker(args):
    seed,arm,steps=args
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        data,rep,source=load_data()
        return fit(seed,0,arm,data,rep,source,steps,True)

def main():
    start=time.time();decision=json.loads((OUT/'decision.json').read_text())
    assert decision['observed_gate_passed']
    paths=[OUT/'protocol.txt',OUT/'rollout.py',OUT/'train.py',OUT/'refit_parallel.py',OUT/'decision.json',V1/'data_splits.npz',V1/'model.py',JOINT/'representation.joblib']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    jobs=[(seed,arm,decision['full_refit_updates'][arm]) for seed in SEEDS for arm in ARMS]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        reports=list(pool.map(worker,jobs))
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('full_training.json',dict(models=reports,input_sha256=hashes,inputs_unchanged=unchanged,
        seconds=time.time()-start,torch_version=torch.__version__,scope='All allowed P3 cells; frozen early representation; no E12.5 truth',
        scheduling='Two independent processes, two CPU threads per fit',training_cells={'E8.5':16787,'E9.5':17057,'E10.5':24979}))
    print('FULL P3 REFITS COMPLETE',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
