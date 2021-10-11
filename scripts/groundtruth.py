import json

with open("../data/preprocessed-data.json") as f:
    content = f.read()

data = json.loads(content)
with open("../data/groundtruth.csv", "w") as output:
    output.write("bug_id,duplicate\n")
    for item in data:
        label = item["dup_id"] if "dup_id" in item else "NO"
        output.write("%s,%s\n" % (item["bug_id"], label))
