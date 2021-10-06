from urllib.request import *

url = "https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id=1626349"
response = urlopen(url)
content = response.read().decode("utf-8", errors="ignore")

with open("1626349.xml", "w") as f:
    f.write(content)


