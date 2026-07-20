import argparse, json
from collections import defaultdict
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int,required=True); p.add_argument("--stratify-field"); a=p.parse_args(); lines=[x for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]
 if a.stratify_field:
  groups=defaultdict(list)
  for line in lines: groups[str(json.loads(line).get(a.stratify_field,"unknown"))].append(line)
  keys=sorted(groups); base=a.limit//len(keys); remainder=a.limit%len(keys); lines=sum((groups[k][:base+(i<remainder)] for i,k in enumerate(keys)),[])
 else: lines=lines[:a.limit]
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(f"wrote {len(lines)}")
if __name__=="__main__": main()
