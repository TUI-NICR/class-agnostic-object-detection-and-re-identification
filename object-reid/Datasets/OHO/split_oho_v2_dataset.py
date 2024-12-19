import csv

# discard all non-tool objects
with open("non_tool_classes.txt", "r") as f:
    non_tool_classes = [c.strip() for c in f]
for part in ["query", "test"]:
    with open("oho_reid_v1/" + part + ".csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [r for r in reader]
    new_data = []
    for e in data:
        if not (e[2] in non_tool_classes):
            new_data.append(e)
    with open("oho_reid_v2/" + part + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_data)
