# Native C550 probes

Standalone MACA C++ runtime probes and a standard-library Python oracle. The
initial toolchain is mxcc 1.0.0, MACA 3.5.3.18. Source availability does not
establish compilation or GPU correctness; retain those results separately.

## Procedure

Keep generated inputs, binaries and results outside the source checkout.
Run Python preparation and compilation before acquiring a GPU lease. Compilation
requires an explicit C550_ARCH from observed compiler/device evidence, so the
compiler does not auto-detect a GPU. The initial inspected compiler reports
xcore1000 as its internal target while the device reports XCORE1002; preserve
that distinction in the run record.

1. python3 experiments/native/native_probe.py prepare /tmp/metax-native-input
2. C550_ARCH=xcore1000 bash experiments/native/compile.sh /tmp/metax-native-probe
3. mkdir /tmp/metax-native-output
4. Acquire and bind exactly one C550 using the installed gpu-infra skill.
5. /tmp/metax-native-probe --run /tmp/metax-native-input /tmp/metax-native-output
6. Release the lease immediately after the device process exits.
7. python3 experiments/native/native_probe.py check /tmp/metax-native-input /tmp/metax-native-output

The program refuses anything other than exactly one visible device whose runtime
name is MetaX C550. It records logical index 0, PCI bus ID and device properties.
Use fresh output directories: an existing raw.jsonl is refused. Incomplete runs
and missing samples, metadata or payloads fail the checker.

## Workload and exact oracle

The authoritative cases.tsv plan contains 49 cases: copy with blocks
64/128/256/512, lengths 1/63/64/65/127/129/511/513/4097/4194317; gather with
1,048,573 outputs, block 256 and strides 1/2/4/8/16; empty one-block kernels
with each block size.

Input i is float32(i), for 0 <= i < 16,777,216. Every input is finite, unique
and exactly representable. Gather uses input[i * stride] without wraparound,
reading n unique indices. CPU checking verifies every input value, every output
bit pattern, finite payload counts and 32 guard words on each side. Guard words
remain uint32 0xffffffff. Unwritten payloads remain nonfinite and fail.

Explicit device buffers are one 64 MiB input and one sequential output, each
output at most 64 MiB including guards. The default largest output is about
16 MiB. Guard checking detects nearby boundary writes, not arbitrary invalid
reads or distant writes. Only final outputs are saved; correctness of every
intermediate repeated launch is not established.

## Timing scope

Each case has 20 warmups followed by synchronization, then 10 samples of 100
launches. All status-returning runtime calls are checked. raw.jsonl retains
each mcEventElapsedTime batch result and host enqueue duration. Allocation,
host-device transfer and file writes are outside the event interval.

Events use the default stream. Their interval can include device idle gaps
while the host submits launches. Host enqueue time includes the per-launch
mcGetLastError query. Dividing by 100 gives a batch mean, not isolated kernel
latency or pure hardware launch overhead. Empty-kernel timing characterizes
this submission protocol.

Addresses repeat. There is no explicit application cache reset; runtime cache
policy is unverified. MACA_LAUNCH_MODE, MACA_LAUNCH_BLOCKING and
MACA_DIRECT_DISPATCH are inherited unchanged and saved as strings or JSON null
when unset. These environment values alone do not establish effective cache
behavior. Do not describe these results as unflushed-cache measurements.

Throughput is logical read-plus-write bytes divided by time, in decimal GB/s.
It is not measured DRAM traffic, DRAM bandwidth or a peak-efficiency claim.
Increasing gather stride changes both input footprint and transaction access;
these timings alone cannot separate cache effects from coalescing effects.
This probe collects no profiler counters.

In the inspected MACA 3.5.3 header, waveSize aliases warpSize. Both recorded
fields therefore represent one API observation, not two independent hardware
confirmations. Kernels assume no wave width or peak performance number.

## CPU verification

Run python3 -m unittest discover -s tests -p test_native_probe.py. These tests
check corrupted input, wrong indices, nonfinite payloads, guard writes, missing
or duplicate timing samples and metadata mismatch on synthetic CPU fixtures.
They do not compile MACA C++ or prove GPU correctness.
