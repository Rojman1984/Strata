# Strata on Radeon Instinct MI25 (gfx900): validation record

Branch `mi25-gfx900` of this fork is upstream `c62d5e3` (v0.2.9) plus three portability commits. It builds and runs on gfx900 through Strata's existing gfx906 (wave64) HIP backend. This page records how the branch was validated (gates 0-7, November 2026), the exact toolchain and binary, the caveats, and the configuration that went into service.

It describes one machine, one model and one ROCm release. It is a field report, not a support statement.

## Hardware and software

| Item | Value |
|---|---|
| GPUs | 3× Radeon Instinct MI25 16 GB HBM2 (gfx900, wave64, ECC on). One on PCIe 3.0 x16; two behind x1 links (~0.7 GB/s measured). |
| Host | Ryzen 5 4600G (AVX2, no AVX-512); 56 GB DDR4-2400 (non-ECC), validated with memtest86+, see "Lessons" |
| ROCm | 6.3.3. AMD clang 18.0.0git (roc-6.3.3); HIP 6.3.42134; hipBLAS 2.3; rocBLAS 4.3.0 (gfx900: tuned FP16/FP32 Tensile, BF16 fallback kernels only); hipBLASLt 0.10 linked |
| ggml / llama.cpp | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (the pin in `third_party/ggml/VERSION.txt`) |
| Model | Qwen3.8-Flash-Next GSQ-RCO **Q2_0** (ISTA-DASLab, `ed59f92`, Strata's pin), packed with `tools/iq_pack.py` |
| MTP draft layer | Built with Strata's own tools: `tools/mtp_fetch.py` (pinned checkpoint revision `de4b8e4d43b9`, all 31 tensors SHA-256-verified against the table in that file), `mtp_pack.py --experts q2_0`, `mtp_rt.py`, default draft vocabulary. `mtp-q2_0.gguf` sha256 `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`. |

## Build

```
cmake -S . -B build -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx900 \
  -DSTRATA_GGML_DIR=<llama.cpp @ 3cf0325> -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build strata
```

**Validated binary:** built from `8576e3f`, 164/164 steps, 0 errors.

| Property | Value |
|---|---|
| Size | 44,786,480 bytes, not stripped |
| sha256 | `0a115b6ffc8b55ac63be9117fd4a622d73fa0990fd5dd500eebc5b551444b35b` |
| Device code | `.hip_fatbin` contains only `gfx900` code objects |

**The build is not bit-reproducible.** clang gives every HIP translation unit a random `__hip_cuid_*`, so a rebuild has the same section sizes but different bytes. `-fuse-cuid=hash` would fix that; it was not used here, to keep the build line unchanged.

## The three commits

These are local fixes. Each has a corresponding upstream fix, still open as of 2026-11-04; those take different approaches and were not the code validated here.

| Commit | Change | Scope |
|---|---|---|
| `ff07c87` | `hip_compat`: map `cudaStreamCreateWithPriority` and `cudaDeviceGetStreamPriorityRange` to their HIP equivalents | A gap in the compat layer. Likely affects every `STRATA_HIP_GFX906` build. Upstream fix: [#4766](https://github.com/Niko1221/Strata/pull/4766), which instead keeps these builds on `dflash2.cpp`'s HIP path (preferred). |
| `610a6e3` | `iq_kernels`: route two `__shared__ alignas(16) int4` tiles through the file's existing #778 macro form | clang 18 (ROCm 6.3) rejects the attribute placement; the CUDA build is unchanged. Upstream fix: [#4600](https://github.com/Niko1221/Strata/pull/4600) |
| `8576e3f` | Prefill GDN out-norm (`STRATA_GDN_NOY=1` path): a portable FP16 store on gfx900 instead of `v_fma_mixlo_f16`, which gfx900 lacks | gfx900 only. Rounds FP32→FP16 twice (up to 1 FP16 ulp vs gfx906); upstream's `gfx906_gdn_rec_head` test is bitwise equal with and without the opt-in path. Upstream fix: [#4983](https://github.com/Niko1221/Strata/pull/4983) |

## Gates

| Gate | Result |
|---|---|
| 0 Library audit | Pass. BF16 GEMMs exist on gfx900 only as rocBLAS fallback kernels. |
| 1 Build | Qualified pass: builds with the three commits above |
| 2 Shipped test suite | Pass with explained exceptions: 68 pass, 5 not applicable to HIP (CUDA/RDNA-only tests), 2 skipped (fixture generator). One HIP runtime finding: the first copy stream + event take 148 MiB of VRAM (the test expects < 16 MiB). |
| 3 Single-card inference | Pass. Coherent output; 15.6 GiB peak VRAM; 0 HIP errors. |
| 4 Reference vs llama.cpp (same GGUF, Vulkan, `--cpu-moe`) | Pass. Step-1 top-5 identical on 3 prompts. Teacher-forced argmax 99.6% / 100% / 98.8%; all 4 disagreements are near-ties (margin ≤ 0.03) where Strata picks llama.cpp's #2. Re-run on validated RAM: identical. |
| 5 Single-card performance | Pass. `STRATA_EXP_MODE=2` beats the gfx906 default 7 by ~2.5% decode. Mode 7 differs from Mode 2 only in its LDS down kernel, which is bit-identical but slower here. |
| 6 Three-card layer split | Pass: correct, capacity only. Bit-deterministic; within tolerance of one card. **The one-GPU hand-off check (`--layer-split K --split-device 0`) is bit-exact only with `--pcie-frac 0`.** With PCIe expert reads on, the two stages hold a different expert mix and rounding differs. |
| 7 Pipeline / overlap / MTP | Pass. `--pipeline-windows 2` is **bit-identical to the serial loop** (the first such check on gfx906-class hardware). MTP and `STRATA_SPLIT_OVERLAP=1` give a measurable benefit; pipelining gives none on this machine. |

Correctness checks compare tokens and top-5 logprobs (`logprobs=5`) against a single-card control. They use a frozen expert cache (`--adapt-every 100000`) and `STRATA_ROUTE_TAIL_SKIP=0`, with tolerances fixed before each run: same top-1, ≥ 4/5 shared top-5, |Δp| ≤ 0.05, and divergence only at near-ties (< 0.10).

## Performance

Serve mode, greedy, 256-token decode, 1,515-token prompt. Median of 5; ranges within ±1%.

| Configuration | Decode tok/s | Prompt tok/s |
|---|---|---|
| One card (x16), no MTP | 20.96 | 156.6 |
| One card, MTP `--spec 2` | 27.94 | 155.3 |
| **One card, MTP `--spec 4 --spec-min-p 0.5`** | **30.33** | 155.4 |
| 3 cards (2 behind x1), no MTP | 10.58 | 137.5 |
| 3 cards + `STRATA_SPLIT_OVERLAP=1` | 17.07 | 138.6 |
| 3 cards + MTP + overlap | 27.84 | 136.2 |
| 3 cards + MTP + `--pipeline-windows 2` | 25.55 | 136.3 |

Draft acceptance: 88% at spec 2, 73% at spec 4 (one card); 68% with pipelining.

The split's cost was mostly host-side synchronization, not transfer. Overlap alone took a decode window from ~94 to ~59 ms.

**In service:** the single x16 MI25 runs MTP `--spec 4 --spec-min-p 0.5`, `STRATA_EXP_MODE=2` and `STRATA_ROUTE_TAIL_SKIP=0`, with the adaptive cache and prompt cache on. That gives **about 28.5 tok/s decode** through an OpenAI-compatible proxy (`serve/server.py`).

Tail skip is off for exact expert math. On this card it gave no measurable speed, and it skipped ~14k experts per request.

## Caveats

- **Run-to-run output.** With the adaptive cache, prompt cache or tail skip on, repeated greedy requests in one long-lived session can differ at near-ties, because cache residency changes the GPU/CPU rounding mix. Even with a frozen cache, the first request after load differs from later ones. **For reproducible benchmarks, use one request per fresh engine.** With tail skip off, that reproduces exactly.
- **Native packs** decode only through speculative verify windows (`--spec ≥ 2`) and cannot dump per-position logits. Use `STRATA_DUMP_FIRST_LOGITS` and `logprobs=K` in serve mode instead.
- **Logs.** `AMD_LOG_LEVEL=1` prints ~20 rocBLAS "Cannot find the function: Cijk…" lookup lines per run. They are lookups across code-object files, not errors.
- **Serialization probes do not work.** `AMD_SERIALIZE_KERNEL=3` / `AMD_SERIALIZE_COPY=3` hang the decode loop: GPU and CPU expert pool exchange work through rings. Don't use them as debugging probes.
- **Startup.** A split over x1 links starts in ~75 s vs ~24 s for one card.
- **Not covered:** contexts beyond 4K, other packs, ROCm releases other than 6.3.3, sampling other than greedy.

## Lessons

- **Test host RAM before debugging an inference engine.** Rare, unique, whole-vector logit perturbations (~1 run in 5; occasional collapse into special tokens) that survived frozen caches, serialization and residency checks turned out to be **one failing DIMM**. memtest86+ found stuck bits at two fixed addresses. With the stick removed: 16/16 runs bit-identical, and the perturbations were gone.
- **A per-run check that catches this kind of fault:**
  - the step-1 logits of a fixed prompt, compared bit for bit with a reference (`STRATA_DUMP_FIRST_LOGITS`);
  - plus `STRATA_VERIFY_COMPLEMENT=1`, which checksums every filled cache slot.
- **`llvm-objcopy --dump-section` with no output file rewrites its input.** Inspecting `.hip_fatbin` that way silently changed the binary's hash. Hash the binary before inspecting it.
