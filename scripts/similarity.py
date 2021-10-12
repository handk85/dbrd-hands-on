from strsimpy.cosine import Cosine
from multiprocessing import Pool
import os
import sys
import json


if len(sys.argv) < 2:
    print("Usage: python crawl.py [PROJECT_NAME]")
    sys.exit(0)
project_name = sys.argv[1].upper()

# Load data from the previous steps
with open("../data/preprocessed-data-%s.json" % project_name) as f:
    content = f.read()
data = json.loads(content)

# The number of recommendations parameter (i.e., n-recommendations)
N = 10
# The number of process
NUM_PROCESS = 8


def concat_title_and_desc(title: str, desc: str):
    return "%s\n\n%s" % (title, desc)


def similarity(title1: str, desc1: str, title2: str, desc2: str):
    # Use default k value
    cosine = Cosine(3)
    s1 = concat_title_and_desc(title1, desc1)
    s2 = concat_title_and_desc(title2, desc2)

    return cosine.similarity(s1, s2)


def map_func(target_item: dict):
    print("PID: %s, BugID: %s" % (os.getpid(), target_item["bug_id"]))
    values = dict()
    for another_item in data:
        # Should not use future data and the same data
        if another_item["bug_id"] >= target_item["bug_id"]:
            continue

        s = similarity(target_item["title"], target_item["description"],
                       another_item["title"], another_item["description"])
        values[another_item["bug_id"]] = s

    return {"bug_id": target_item["bug_id"], "values": [[k, values[k]] for k in
                                                        sorted(values, key=values.get, reverse=True)[:N]]}


if __name__ == "__main__":
    with Pool(processes=NUM_PROCESS) as p:
        results = p.map(map_func, data)

    with open("../data/similarity-%s-%s.csv" % (project_name, N), "w") as f:
        f.write("BugId,Recommendation,Similarity\n")
        for item in results:
            f.write("%s,%s,%s\n" % (item["bug_id"], item["values"][0], item["values"][1]))
