B=https://data.binance.vision/data/spot/monthly/klines
for s in BTCUSDT ETHUSDT LTCUSDT PIVXUSDT QKCUSDT GNOUSDT; do mkdir -p spot_1m/$s
for m in 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06; do
 f=spot_1m/$s/$s-1m-$m.zip; [ -s $f ] || curl -sf -o $f $B/$s/1m/$s-1m-$m.zip || { rm -f $f; echo miss $s $m; }
done; done; echo ALLDONE
