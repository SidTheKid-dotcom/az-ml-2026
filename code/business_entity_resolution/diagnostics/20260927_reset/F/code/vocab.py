import re, collections, unicodedata
comp_last = collections.Counter(); comp_all = collections.Counter(); after_num = collections.Counter(); first_tok = collections.Counter(); prefix = collections.Counter(); legal = collections.Counter()
def fold(s): return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().lower()
for line in open("/workspace/az-ml-new/reset/F/fr_all.tsv", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 4: continue
    eid, name, addr = p[0], p[1], p[2]
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    for c in comps:
        if not re.search(r"\d", c): comp_all[fold(c)] += 1
        m = re.match(r"^\s*(?:n\s*[°º]|no\.?|#|\()?\s*\(?(\d+)\)?\s*(?:bis|ter|quater|[a-z])?\b\s*[-,]?\s*([^\s\d]+)", c, re.I)
        if m: after_num[m.group(2)] += 1
        m2 = re.match(r"^\s*([^\s\d]+)", c)
        if m2 and not re.search(r"\d", c.split()[0]): first_tok[m2.group(1)] += 1
        m3 = re.match(r"^\s*(\S*?)\s*\d", c)
        if m3 and m3.group(1): prefix[m3.group(1)] += 1
    for t in re.findall(r"[\w.]+", name):
        tt = t.replace(".", "").lower()
        if tt in {"sarl","sas","sasu","sa","sci","eurl","ei","snc","sàrl","selarl","sca","scop","gie","scm","scp","selas","earl","gaec","sem"}: legal[t] += 1
print("LEGAL", legal.most_common(60))
print("PREFIX", prefix.most_common(40))
print("AFTERNUM", after_num.most_common(150))
print("FIRSTTOK", first_tok.most_common(120))
print("NODIGIT_COMPONENTS", len(comp_all), comp_all.most_common(80))
