set -u
B=https://data.binance.vision/data
for s in BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT LTCUSDT LINKUSDT AVAXUSDT; do
 for y in 2021 2022 2023 2024 2025 2026; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  [ "$y$m" -gt 202609 ] && continue
  for kind in "spot/monthly/klines/$s/1h/$s-1h-$y-$m" "futures/um/monthly/klines/$s/1h/$s-1h-$y-$m" "futures/um/monthly/fundingRate/$s/$s-fundingRate-$y-$m"; do
   d=$(echo $kind | cut -d/ -f1-3 | tr / _); mkdir -p $d/$s; f=$d/$s/$(basename $kind).zip
   [ -s $f ] || curl -sf -o $f "$B/$kind.zip" || rm -f $f
  done; done; done; echo done $s; done
echo ALLDONE
