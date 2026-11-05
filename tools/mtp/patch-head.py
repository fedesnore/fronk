"""Exact single-token HC activation reuse, only in a staged engine."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
source = root / "ds4_metal.m"
s = source.read_text()

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise RuntimeError("Unexpected head patch anchor: " + old[:80])
    s = s.replace(old, new)

replace("    QWEN4_K_HC_GATE_MIX_PAIR_Q8,",
        "    QWEN4_K_HC_GATE_MIX_PAIR_Q8,\n    FRONK_K_HC_SINGLE,")
replace('    "kernel_qwen4_hc_gate_mix_pair_q8",',
        '    "kernel_qwen4_hc_gate_mix_pair_q8",\n    "kernel_fronk_hc_mix_f16_1",')
replace("int ds4_gpu_qwen4_hc_gate_mix_tensor(", """/* The low-rank sigmoid factors are identical for every output coordinate.
 * Stage them once per threadgroup, keeping the original per-lane multiply
 * order. This flag is initialized once; no new allocation or CPU/GPU fence. */
static int g_fronk_head_hc = -1;
void ds4_gpu_fronk_head_hc(int enabled) { g_fronk_head_hc = enabled != 0; }
int ds4_gpu_qwen4_hc_gate_mix_tensor(""")
anchor = '    const bool pair = n_tokens == 2u'
at = s.index(anchor, s.index("int ds4_gpu_qwen4_hc_gate_mix_tensor("))
s = s[:at] + """    if (g_fronk_head_hc<0) {
        const char *mode=getenv("FRONK_HEAD_HC");
        /* Keep opt-in until full-chat throughput beats both controls. */
        g_fronk_head_hc=mode && !strcmp(mode,"1");
    }
    if (g_fronk_head_hc && weight_type==1u && n_tokens==1u && n_rank<=1024u)
        return qwen4_dispatch(FRONK_K_HC_SINGLE,&args,sizeof(args),b,4,
            MTLSizeMake((n_embd+3u)/4u,1,1),MTLSizeMake(128,1,1),
            (NSUInteger)n_rank*2u*sizeof(float));
""" + s[at:]
source.write_text(s)
metal = root / "metal/qwen4.metal"
if "FRONK_HC_INSTANCE(1)" not in metal.read_text():
    metal.write_text(metal.read_text() + "\n" + (Path(__file__).parent / "fronk_dense.metal").read_text())
header = root / "ds4_gpu.h"
header.write_text(header.read_text() + "\nvoid ds4_gpu_fronk_head_hc(int enabled);\n")
