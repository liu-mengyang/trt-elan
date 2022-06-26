import os
import re
from pprint import pprint
root = 'plans'
data = {}
for path in os.listdir(root):
    match = re.findall(r"elan_x4_to_fp32_Conv_([0-9]+).plan.txt", path, re.S)
    if match == None or len(match) <= 0:
        continue
    name = int(match[0])
    with open(os.path.join(root, path), "r") as f:
        data[name] = [float(s.strip()) for s in str(f.readline()).split(",")]
        
pprint(data)

mea = []
for k,v in data.items():
    mea.append((k,v[-1]))

from operator import itemgetter
mea = sorted(mea, key=itemgetter(1))
pprint(mea)

for i in mea:
    print("%d\t%f"%(i[0],i[1]))