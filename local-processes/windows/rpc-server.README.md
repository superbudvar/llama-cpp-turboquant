# rpc-server.bat - launcher for the Windows RPC peer

Runs `ggml-rpc-server.exe` with a timestamped per-run log, and turns the
connect/close spam into a counter.

## Why the output needs help

The server prints two lines per client connection:

```
Accepted client connection
Client connection closed
```

The fit solver opens one connection per probe (up to 32), so a single model load
produces a wall of identical lines that buries the messages that matter
(`CUDA graph warmup complete`, allocation errors, and so on).

## What the script does

- **Logs every run** to `logs\rpc-server-<stamp>.log`, next to the exe. One file
  per run, never overwritten. The log header records the command line, the
  exe's SHA256 and the `GGML_TQ_NATIVE` value, so a log identifies its own
  build.
- **Timestamps the console output** (`HH:mm:ss.fff`).
- **Counts the connections** instead of printing the pairs:
  ```
  [21:57:12.480] conn #1 accepted
  [21:57:12.480] conn #1 closed (0 still open)
  [21:57:12.612] ggml_backend_cuda_graph_compute: CUDA graph warmup complete
  ```
  Everything that is not a connect/close line is shown verbatim.
- **Keeps the raw text** in the log, unfiltered, so nothing is lost.
- **Sets `GGML_TQ_NATIVE=1` by default**, so TQ4_1S weights stay native on this
  peer. The header prints the value, so a log says which way it went (see
  below).

## Usage

```bat
rpc-server.bat                    run with the defaults
rpc-server.bat -p 50052           override the port
rpc-server.bat --tq-native=0      expand TQ4_1S to q8_0 instead (fast prefill)
rpc-server.bat --foreground       raw output, no connection counter
rpc-server.bat --help
```

Defaults: host `0.0.0.0`, port `50052`, log dir `logs\` next to the script,
`GGML_TQ_NATIVE=1`.

The script looks for `ggml-rpc-server.exe` **next to itself**, so drop it into
`F:\llama-build` beside the built `bin\` contents. Edit `RPC_DIR` at the top if
the exe lives elsewhere.

Stop with Ctrl-C.

## `GGML_TQ_NATIVE` and the TQ4_1S expansion

TQ4_1S is a WHT-rotated 4-bit type. A generic GEMM (cuBLAS) has to apply the
inverse Walsh-Hadamard per element, which is slow per tile, so by default the
server dequantises and un-rotates the whole tensor **once** at load time into
plain q8_0. Every GEMM after that is an ordinary fast q8_0 matmul.

| `GGML_TQ_NATIVE` | what the server does | cost |
|---|---|---|
| unset | expands TQ4_1S to q8_0 | fast prefill, **1.7x the VRAM** |
| `0` | expands TQ4_1S to q8_0 | same as unset |
| `1` (**this script's default**) | keeps the native 5 bpw bytes | **1.7x less VRAM**, 2x slower prefill |

Native TQ4_1S decode is 29-33% faster than the converted path, so the
conversion default is usually the wrong trade - it spends the scarcest
resource to buy prefill speed. The fork's own note says to set
`GGML_TQ_NATIVE=1` for any TQ4_1S model.

**Why this peer especially:** the card has **8191 MiB** of VRAM, and the 35B-A3B
MoE already puts 6.0-6.2 GiB of weights on it (in IQ4_XS). The q8_0 expansion
costs 1.7x, so the same share would need 10.2-10.5 GiB - more than the card
has. The expanded path cannot hold the MoE at all, whatever the split.

Native TQ4_1S is 5.0 bpw against IQ4_XS's 4.25 bpw, so the peer's share would
grow by that ratio (~7.2 GiB on the same split). That is tighter than the
expanded path by a wide margin, and it is the reason native is the default
rather than a preference. Whether a TQ4_1S MoE rung fits at all is a question
for the build, not for the flag.

The price is 2x slower prefill on the peer, because cuBLAS dequant-to-f16 needs
a per-element inverse WHT. That is the right way round for a spill device: the
peer exists to hold weights and decode, and it cannot hold them if they are
expanded.

Note the flag is read by the **server** process, so it has to be set here, on
the Windows box, not on the Linux client. The client's own `GGML_TQ_NATIVE` has
no effect on the weights that live on the peer - the client does not hold
them.

## Related

The TQ4_1S-over-RPC type mismatch (the `!!!!!!!!!!!!!!` symptom) is a **client**
bug, fixed by PR #405 in `ggml/src/ggml-rpc/ggml-rpc.cpp` - the client has to
re-type the tensor when the server converted it. The Windows build does not
need that fix; it is the server side. But the Linux client must be on a build
that has it.

With `GGML_TQ_NATIVE=1` on the peer the mismatch cannot arise at all: the
server keeps the TQ4_1S bytes, so the client's retyping test (which fires only
when the peer reports a q8_0-sized allocation) never triggers.
