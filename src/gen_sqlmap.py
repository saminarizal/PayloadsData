"""Assemble concrete SQLi strings from sqlmap's own test templates (data/xml/payloads/*.xml)
and boundaries (data/xml/boundaries.xml), following lib/controller/checks.py and lib/core/agent.py
(prefixQuery/suffixQuery). Strings are only generated for offline classification; nothing is sent."""
import xml.etree.ElementTree as ET, glob, random, re, sys, csv, os
X = sys.argv[1]; OUT = sys.argv[2]
rng = random.Random(42)
def rint(n=4): return str(rng.randint(10**(n-1), 10**n-1))
def rstr(n=4): return ''.join(rng.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(n))
def rng_list(s):
    out=set()
    for part in (s or '').split(','):
        part=part.strip()
        if not part: continue
        if '-' in part: a,b=part.split('-'); out |= set(range(int(a),int(b)+1))
        else: out.add(int(part))
    return out
def fill(s):
    if s is None: return ''
    s = s.replace('[GENERIC_SQL_COMMENT]', '-- [RANDSTR]')
    for k in ['[RANDNUM5]','[RANDNUM4]','[RANDNUM3]','[RANDNUM2]','[RANDNUM1]','[RANDNUM]']:
        while k in s: s = s.replace(k, rint(), 1)
    while '[RANDSTR]' in s: s = s.replace('[RANDSTR]', rstr(), 1)
    s = s.replace('[SLEEPTIME]', '5').replace('[ORIGVALUE]', '1')
    s = s.replace('[DELIMITER_START]', 'q'+''.join(rng.choice('zxjkvbp') for _ in range(3))+'q')
    s = s.replace('[DELIMITER_STOP]', 'q'+''.join(rng.choice('zxjkvbp') for _ in range(3))+'q')
    return s
B=[]
for b in ET.parse(os.path.join(X,'boundaries.xml')).getroot().findall('boundary'):
    B.append(dict(level=int(b.findtext('level')), clause=rng_list(b.findtext('clause')), where=rng_list(b.findtext('where')),
                  prefix=b.findtext('prefix') or '', suffix=b.findtext('suffix') or ''))
rows=[]
STYPE={1:'boolean-based blind',2:'error-based',3:'inline query',4:'stacked queries',5:'time-based blind',6:'UNION query'}
for f in sorted(glob.glob(os.path.join(X,'payloads','*.xml'))):
    for t in ET.parse(f).getroot().findall('test'):
        st=int(t.findtext('stype')); lvl=int(t.findtext('level')); risk=int(t.findtext('risk'))
        tcl=rng_list(t.findtext('clause')); twh=rng_list(t.findtext('where'))
        req=t.find('request'); pay=req.findtext('payload'); com=req.findtext('comment')
        if st==6:
            ch=req.findtext('char'); bodies=[]
            for n in (1,3,5,10):
                c=ch if ch and ch!='[CHAR]' else '[RANDNUM]'
                bodies.append('UNION ALL SELECT '+','.join([c]*n))
        else:
            bodies=[pay]
        for body in bodies:
            for b in B:
                if not (0 in tcl or 0 in b['clause'] or tcl & b['clause']): continue
                for w in sorted(twh):
                    if w in (1,2) and w not in b['where']: continue
                    orig = '1' if w==1 else ('-'+rint() if w==2 else '')
                    if w==3: q=''
                    elif st==4: q=b['prefix']
                    else:
                        q=b['prefix']
                        if not (body and body[0]==';') and not (q and q[-1] in '()' and body and body[0] in '()') and not (q and q[-1]=='('): q+=' '
                    expr=q+body
                    if com: expr+=com
                    elif w!=3 and b['suffix']:
                        if re.search(r'\w\Z',expr) and re.search(r'\A\w',b['suffix']): expr+=' '
                        expr+=b['suffix']
                    expr=re.sub(r';\W*;',';',expr)
                    rows.append(dict(technique=STYPE[st], level=max(lvl,b['level']), risk=risk, title=t.findtext('title'), payload=fill(orig+expr)))
                if 3 in twh: break  # REPLACE ignores boundaries: one instance is enough
with open(OUT,'w',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(len(rows))
