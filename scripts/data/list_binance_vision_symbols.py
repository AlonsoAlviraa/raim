import re,urllib.request,sys
def lst(prefix):
    out=[];marker=""
    while True:
        u=f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix={prefix}&delimiter=/&marker={marker}"
        t=urllib.request.urlopen(u).read().decode()
        ps=re.findall(r"<Prefix>([^<]+)</Prefix>",t)[1:]
        out+= [p.rstrip('/').split('/')[-1] for p in ps]
        m=re.search(r"<NextMarker>([^<]+)</NextMarker>",t)
        if not m: break
        marker=m.group(1)
    return out
import json
spot=lst("data/spot/monthly/klines/"); perp=lst("data/futures/um/monthly/klines/")
bad=re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
stable={"USDCUSDT","BUSDUSDT","TUSDUSDT","FDUSDUSDT","USDPUSDT","DAIUSDT","EURUSDT","GBPUSDT","AEURUSDT","PAXUSDT","USTUSDT","USDSUSDT","USDSBUSDT","SUSDUSDT","EURIUSDT","XUSDUSDT","USD1USDT","BFUSDUSDT","PAXGUSDT","AUDUSDT","BKRWUSDT","RLUSDUSDT"}
s=[x for x in spot if x.endswith("USDT") and not bad.search(x) and x not in stable]
p=[x for x in perp if x.endswith("USDT") and x not in stable]
json.dump({"spot":s,"perp":p},open("symbols.json","w"))
print(len(spot),len(s),len(perp),len(p))
