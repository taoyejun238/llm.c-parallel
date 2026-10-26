#!/bin/bash
# Compute vs NCCL-kernel overlap per training phase, from Nsight Systems traces of rank 0.
# Usage: ./analyze.sh d48_z1_rank0 d48_z2_rank0 ...   (nsys-rep files in this directory)
# Uses the last 4 training steps; phases are the NVTX ranges gpt2_forward / gpt2_backward_and_reduce /
# gpt2_update (each ends with a device sync, so the host range contains the step's GPU work).
cd "$(dirname "$0")"
for P in "$@"; do
  DB=$P.sqlite
  [ -f $DB ] || nsys export --type sqlite --force-overwrite true -o $DB $P.nsys-rep > /dev/null
  NAME="coalesce(n.text,s.value)"
  FROM="NVTX_EVENTS n left join StringIds s on n.textId=s.id"
  for R in gpt2_forward gpt2_backward_and_reduce gpt2_update; do
    if [ $R = gpt2_forward ]; then  # the training forward right before each of the last 4 backwards
      RANGES=$(sqlite3 -separator ' ' $DB "with b as (select n.start st from $FROM where $NAME='gpt2_backward_and_reduce' order by n.start desc limit 4) select (select max(n.start) from $FROM where $NAME='gpt2_forward' and n.start < b.st), (select max(n.end) from $FROM where $NAME='gpt2_forward' and n.start < b.st) from b;")
    else
      RANGES=$(sqlite3 -separator ' ' $DB "select n.start, n.end from $FROM where $NAME='$R' order by n.start desc limit 4;")
    fi
    echo "$RANGES" | while read ST EN; do
      sqlite3 -separator ' ' $DB "select k.start, k.end, case when s.value like 'nccl%' then 1 else 0 end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.start >= $ST and k.start <= $EN order by k.start;" | awk -f overlap.awk
    done | awk -v p=$P -v r=$R '{c+=$3; m+=$8; o+=$12; x+=$20} END{printf "%-14s %-26s per step: compute %6.1f ms, NCCL %5.1f ms, overlapped %5.1f ms, exposed NCCL %5.1f ms\n", p, r, c/4, m/4, o/4, x/4}'
  done
done
