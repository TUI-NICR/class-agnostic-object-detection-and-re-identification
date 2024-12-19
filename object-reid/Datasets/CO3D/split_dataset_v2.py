"""
I do not remember what this script does. I believe it filters the IDs which were removed when co3d_reid_v2 was created.
"""
import csv
import os


with open("co3d_reid_v2.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data_v2 = [r for r in reader]

splits = []
for split_name in ["train", "test", "query"]:
    with open(f"co3d_reid_v1/{split_name}.csv", "r") as f:
        reader = csv.reader(f)
        header_ = next(reader)
        assert header == header_
        splits.append([r for r in reader])

for i, split in enumerate(splits):
    d = {e[0] for e in split}
    assert len(d) == len(split)
    splits[i] = d

new_splits = [[header], [header], [header]]
for entry in data_v2:
    for i, split in enumerate(splits):
        if entry[0] in split:
            new_splits[i].append(entry)

print(*[len(split) for split in new_splits])
print(*[len(split) for split in splits])
assert sum([len(split) for split in new_splits]) == len(data_v2) + 3

id_t, id_q = set(), {}
for t in new_splits[1][1:]:
    id_t.add(t[2])
for i, q in enumerate(new_splits[2][1:]):
    id_ = q[2]
    if id_ not in id_q:
        id_q[id_] = []
    id_q[id_].append(i)
rem = set()
for id_, inds in id_q.items():
    if id_ not in id_t:
        rem.update(inds)
mod_q = [header]
for i, entry in enumerate(new_splits[2][1:]):
    if i not in rem:
        mod_q.append(entry)
new_splits[2] = mod_q

print(*[len(split) for split in new_splits])

os.makedirs("co3d_reid_v2", exist_ok=True)
for split_name, split in zip(["train", "test", "query"], new_splits):
    with open(f"co3d_reid_v2/{split_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(split)
