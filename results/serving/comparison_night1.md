| Cell | A (official MXFP4) | B (BF16) | D-hybrid (attn NVFP4) |
|------|---|---|---|
| decode_in128_c1 | 0.023s / 305t/s | 0.031s / 265t/s | 0.032s / 256t/s |
| decode_in128_c32 | 4.205s / 1254t/s | 0.891s / 1717t/s | 0.917s / 1695t/s |
| decode_in128_c64 | 0.164s / 3674t/s | 0.165s / 1993t/s | 0.185s / 1901t/s |
| decode_in128_c8 | 0.043s / 1479t/s | 0.055s / 819t/s | 0.052s / 773t/s |
| mixed_in1024_c1 | 0.037s / 314t/s | 0.045s / 256t/s | 0.050s / 250t/s |
| mixed_in1024_c32 | 0.356s / 3268t/s | 0.412s / 1851t/s | 0.474s / 1868t/s |
| mixed_in1024_c64 | 0.347s / 3311t/s | 0.404s / 1890t/s | 0.456s / 1859t/s |
| mixed_in1024_c8 | 0.157s / 1055t/s | 0.111s / 735t/s | 0.140s / 774t/s |
| mixed_in8192_c1 | 0.150s / 259t/s | 0.167s / 219t/s | 0.202s / 208t/s |
| mixed_in8192_c32 | 1.821s / 1347t/s | 2.187s / 1011t/s | 2.657s / 911t/s |
| mixed_in8192_c64 | 1.856s / 1324t/s | 2.182s / 1005t/s | 2.661s / 906t/s |
| mixed_in8192_c8 | 0.553s / 880t/s | 0.649s / 600t/s | 0.800s / 567t/s |
| prefill_in1024_c1 | 0.038s / 19t/s | 0.044s / 17t/s | 0.048s / 16t/s |
| prefill_in1024_c32 | 0.397s / 47t/s | 0.373s / 49t/s | 0.425s / 40t/s |
| prefill_in1024_c8 | 1.076s / 12t/s | 0.145s / 45t/s | 0.134s / 43t/s |
| prefill_in8192_c1 | 0.147s / 5t/s | 0.170s / 5t/s | 0.195s / 4t/s |
| prefill_in8192_c32 | 1.715s / 8t/s | 1.998s / 7t/s | 2.418s / 6t/s |
| prefill_in8192_c8 | 0.902s / 8t/s | 1.047s / 7t/s | 1.277s / 6t/s |