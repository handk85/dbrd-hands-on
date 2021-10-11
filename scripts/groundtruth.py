import sys
import json

if len(sys.argv) < 2:
    print("Usage: python crawl.py [PROJECT_NAME]")
    sys.exit(0)
project_name = sys.argv[1].upper()

with open("../data/preprocessed-data-%s.json" % project_name) as f:
    content = f.read()

data = json.loads(content)
with open("../data/groundtruth-%s.csv" % project_name, "w") as output:
    output.write("bug_id,duplicate\n")
    for item in data:
        label = item["dup_id"] if "dup_id" in item else "NO"
        output.write("%s,%s\n" % (item["bug_id"], label))
