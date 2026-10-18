import hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'SHA256SUMS.json').read_text(encoding='utf-8'))['files'];bad=[]
for n,h in m.items():
 p=(r/n).resolve()
 if not p.is_relative_to(r) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(n)
print(json.dumps({'checked':len(m),'mismatches':bad,'scope':'published bytes only'}));sys.exit(bool(bad))
