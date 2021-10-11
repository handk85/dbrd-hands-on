import configparser
import glob
import json
import logging
import sys
import re

regexps = {"bug_id": re.compile("<bug_id>(.*?)</bug_id>"),
        "title": re.compile("<short_desc>(.*?)</short_desc>"),
        "status": re.compile("<bug_status>(.*?)</bug_status>"),
        "resolution": re.compile("<resolution>(.*?)</resolution>"),
        "description": re.compile("<thetext>(.*?)</thetext>", re.MULTILINE | re.DOTALL)}
# dup_id is not a mandatory field in a bug report
rg_dup_id = re.compile("<dup_id>(.*?)</dup_id>")


if len(sys.argv) < 2:
    print("Usage: python crawl.py [PROJECT_NAME]")
    sys.exit(0)
project_name = sys.argv[1].upper()

config = configparser.ConfigParser()
config.read("settings.ini")

if project_name not in config:
    print("No configuration for %s in settings.ini."%project_name)
    sys.exit(1)

OUTPUT_DIR = config[project_name]['output_dir']

data = []
for f_name in glob.glob("%s/*.xml" % OUTPUT_DIR):
    # Read first 1000 lines from the file
    with open(f_name) as f:
        content = "\n".join([f.readline() for x in range(1000)])

    values = dict()
    for key, regexp in regexps.items():
        temp_value = regexp.findall(content)
        values[key] = temp_value[0] if temp_value else ""
    if values["bug_id"] == "":
        logging.info("%s is not a valid bug report" % f_name)
        continue
    values["bug_id"] = int(values["bug_id"])

    # Skip unresolved bug reports
    if "resolution" not in values or values["resolution"] == "" or \
            values["resolution"] == "INVALID" or values["resolution"] == "INCOMPLETE":
        continue

    dup_id = rg_dup_id.findall(content)
    if dup_id:
        values["dup_id"] = dup_id[0]

    data.append(values)

with open("../data/preprocessed-data-%s.json" % project_name, "w") as f:
    f.write(json.dumps(data))
