"""Reproduce this experiment in order; touches only this experiment's outputs."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def run(name,*args):
    subprocess.run([sys.executable,str(HERE/name),*args],cwd=ROOT,check=True)

if __name__=='__main__':
    run('verify.py')
    run('train.py')
    run('evaluate.py')
    run('audit_outputs.py')
    decision=json.loads((HERE/'decision.json').read_text())
    if decision['observed_gate_passed']:
        run('refit_parallel.py')
        run('export_e125.py')
        run('audit_composition.py')
    run('diagnose_mass.py')
    run('summarize.py')
