import re

rg_bug_id = re.compile("<bug_id>(.*?)</bug_id>")
rg_short_desc = re.compile("<short_desc>(.*?)</short_desc>")
rg_bug_status = re.compile("<bug_status>(.*?)</bug_status>")
rg_resolution = re.compile("<resolution>(.*?)</resolution>")
rg_dup_id = re.compile("<dup_id>(.*?)</dup_id>")
rg_description = re.compile("<thetext>(.*?)</thetext>", re.MULTILINE | re.DOTALL)

# Read first 1000 lines from the file
with open("1626349.xml") as f:
    content = "\n".join([f.readline() for x in range(1000)])

bug_id = rg_bug_id.findall(content)[0]
title = rg_short_desc.findall(content)[0]
status = rg_bug_status.findall(content)[0]
resolution = rg_resolution.findall(content)[0]
dup_id = rg_dup_id.findall(content)[0]
# First comment is the description of the bug report
description = rg_description.findall(content)[0]

print({bug_id, status, resolution, dup_id, title, description})

