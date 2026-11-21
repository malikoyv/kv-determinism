#!/usr/bin/bash
# usage: run_mp.sh ARM TAG  (ARM d = L1 only, s = L1 + fs_native L2 with skip_l1, O_DIRECT)
D=$HOME/claude-lab/det; cd $D; . ${VENV:-venv}/bin/activate; mkdir -p res logs kvdisk
ARM=$1; TAG=$2; PORT=17555; HPORT=17556
export LMCACHE_DISABLE_BANNER=1 MPPORT=$PORT VLLM_USE_FLASHINFER_SAMPLER=0
python - <<PY || { echo "STATUS $TAG FAIL port busy" >> STATUS_mp; exit 1; }
import socket,sys
s=socket.socket(); sys.exit(0 if s.connect_ex(("127.0.0.1",$PORT)) else 1)
PY
if [ $ARM = d ]; then
  EXTRA="--l1-size-gb 8"
else
  rm -rf kvdisk/$TAG; mkdir -p kvdisk/$TAG
  EXTRA="--l1-size-gb 4 --l2-store-policy skip_l1 --l2-adapter {\"type\":\"fs_native\",\"base_path\":\"$D/kvdisk/$TAG\",\"num_workers\":4,\"use_odirect\":true}"
fi
lmcache server --port $PORT --http-port $HPORT --chunk-size ${CHUNK:-256} --max-workers 8 --eviction-policy LRU $EXTRA > logs/$TAG.server.log 2>&1 &
SPID=$!
for i in $(seq 60); do python -c "import socket,sys;s=socket.socket();sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" && break; sleep 2; done
ds() { awk '$3=="nvme0n1"{print "read_sectors",$6,"write_sectors",$10}' /proc/diskstats; }
echo "before: $(ds)" > logs/$TAG.disk
python gen.py $ARM res/$TAG.json > logs/$TAG.log 2>&1; rc=$?
echo "after:  $(ds)" >> logs/$TAG.disk; du -sh kvdisk/$TAG >> logs/$TAG.disk 2>/dev/null
kill $SPID; wait $SPID 2>/dev/null; sleep 2
echo "STATUS $TAG rc=$rc $(date)" >> STATUS_mp
