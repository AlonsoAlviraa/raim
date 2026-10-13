import json,re,urllib.request
from concurrent.futures import ThreadPoolExecutor
p=json.load(open('symbols.json'))['perp']
def keys(prefix):
    out=[];marker=""
    while True:
        t=urllib.request.urlopen(f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix={prefix}&marker={marker}",timeout=30).read().decode()
        ks=re.findall(r"<Key>([^<]+\.zip)</Key>",t); out+=ks
        if "<IsTruncated>true" not in t or not ks: return out
        marker=ks[-1]
def job(s):
    try: return keys(f"data/futures/um/monthly/klines/{s}/1d/")+keys(f"data/futures/um/monthly/fundingRate/{s}/")
    except Exception as e: return []
with ThreadPoolExecutor(32) as ex: res=list(ex.map(job,p))
with open('urls_edge.txt','w') as o:
    for ks in res:
        for k in ks:
            ym=re.search(r"(\d{4}-\d{2})\.zip",k).group(1)
            if ym>"2026-09": continue
            s=k.split("/")[5]; kind='perp_1d' if '/klines/' in k else 'funding_all'
            o.write(f"https://data.binance.vision/{k} {kind}/{s}/{k.split('/')[-1]}\n")
print(sum(map(len,res)))
