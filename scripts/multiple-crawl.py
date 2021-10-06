from urllib.request import *

START_INDEX=1626349

URL_TEMPLATE = "https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id=%s"

for i in range(START_INDEX, 0, -1):
    url = URL_TEMPLATE % i
    print(url)

    response = urlopen(url)
    content = response.read().decode("utf-8", errors="ignore")
    
    # You need to create "bug_reports" directory in advance
    with open("bug_reports/%s.xml" % i, "w") as f: 
        f.write(content)


