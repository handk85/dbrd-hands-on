import re
import json

regexps = {"bug_id": re.compile("<bug_id>(.*?)</bug_id>"),
        "title": re.compile("<short_desc>(.*?)</short_desc>"),
        "status": re.compile("<bug_status>(.*?)</bug_status>"), 
        "resolution": re.compile("<resolution>(.*?)</resolution>"),
        "description": re.compile("<thetext>(.*?)</thetext>", re.MULTILINE | re.DOTALL)}
# dup_id is not a mandatory field in a bug report
rg_dup_id = re.compile("<dup_id>(.*?)</dup_id>")

# Read first 1000 lines from the file
with open("1626349.xml") as f:
    content = "\n".join([f.readline() for x in range(1000)])

values = dict()
for key, regexp in regexps.items():
    values[key] = regexp.findall(content)[0]
    dup_id = rg_dup_id.findall(content)
    if dup_id:
        values["dup_id"] = dup_id[0]

print(json.dumps(values))

