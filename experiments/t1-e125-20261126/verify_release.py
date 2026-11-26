"""Verify release files with the Python standard library; optionally check raw inputs."""
import argparse, hashlib
from pathlib import Path

def verify(root,manifest):
    count=0
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        expected,name=line.split('  ',1);p=root/name
        if not p.is_file():raise SystemExit('Missing: '+str(p))
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise SystemExit('Checksum mismatch: '+str(p))
        count+=1
    print('Verified',count,'files:',manifest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path,nargs='?',default=Path.cwd());ap.add_argument('--data',action='store_true');args=ap.parse_args()
    root=args.root.resolve();verify(root,root/'SHA256SUMS')
    if args.data:verify(root,root/'raw_data.sha256')
