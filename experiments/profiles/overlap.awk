# input: start end is_comm  (ns), sorted by start. Prints busy/union stats in ms.
{ s[NR]=$1; e[NR]=$2; c[NR]=$3 }
END {
  # union lengths for comp, comm, and all
  for (k = 0; k < 3; k++) {
    tot = 0; cs = -1; ce = -1
    for (i = 1; i <= NR; i++) {
      if (k == 0 && c[i] != 0) continue
      if (k == 1 && c[i] != 1) continue
      if (cs < 0) { cs = s[i]; ce = e[i]; continue }
      if (s[i] <= ce) { if (e[i] > ce) ce = e[i] } else { tot += ce - cs; cs = s[i]; ce = e[i] }
    }
    if (cs >= 0) tot += ce - cs
    u[k] = tot
  }
  comp = u[0]/1e6; comm = u[1]/1e6; all = u[2]/1e6; ov = comp + comm - all
  printf "compute busy %.1f ms | comm busy %.1f ms | overlapped %.1f ms (%.0f%% of comm) | exposed comm %.1f ms | GPU busy %.1f ms\n", comp, comm, ov, (comm>0?100*ov/comm:0), comm-ov, all
}
