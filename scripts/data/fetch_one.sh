u=$1; o=$2; [ -s "$o" ] && exit 0; mkdir -p $(dirname $o); curl -sf --retry 2 -o "$o" "$u" || rm -f "$o"
