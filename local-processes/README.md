# Local build and VM processes

This directory holds personal, host-specific processes that are intentionally
separate from upstream product scripts. Cherry-pick it as a unit when useful.
Do not put private keys or credentials here.

## Windows CUDA VM build

The host runner starts or reuses the configured Libvirt domain, discovers its
current IPv4 address, verifies the mapped source revision and virtiofs tag,
attaches a temporary qcow2 scratch disk, invokes the Windows build script over
SSH, exports binaries and logs to the host-backed artifact folder, then detaches
and removes the scratch disk.

Run a read-only preflight first:

```sh
python3 local-processes/build_vm.py \
  --profile local-processes/profiles/llama-cpp-turboquant.json --dry-run
```

Run the build only when no other agent is using the VM:

```sh
python3 local-processes/build_vm.py \
  --profile local-processes/profiles/llama-cpp-turboquant.json
```

The profile pins the SSH host-key fingerprint and identity-file path. The runner
does not accept or save an unknown key automatically. For a bridged VM without
QEMU guest agent, it falls back to Libvirt's DHCP-lease and ARP address sources.
It caches the last verified address under the host scratch root and re-verifies
both the pinned key and guest hostname/commit before using it. If discovery and
the cached address both fail, configure the guest agent or a DHCP reservation;
the runner does not scan the subnet.

The qcow2 scratch image is a transient data disk, not an overlay of the VM's raw
system disk. It lives under `/mnt/build/local-processes`; the runner checks free
space and grants the `qemu` group access only to its scratch directory/file.
Windows initializes the new disk only after matching its unique serial. Build
trees, temp files, and logs go there; only the requested executables, DLLs, and a
manifest are copied to the host-backed artifact directory. On a successful run,
the runner detaches and removes the scratch image. If export or detach fails, it
keeps the image and prints its path for recovery.

Project profiles declare the host source root, guest mapping, VM, toolchain,
scratch size, build targets, and artifact paths. Add a new profile rather than
editing the runner for each repository. The current `win11` profile expects the
`turboquant` virtiofs tag to map to this repository and an 8 GiB scratch disk;
the runner refuses to start if host free space cannot cover the configured
scratch size plus its reserve.
The current domain has no QEMU guest-agent channel, so IP discovery uses lease/ARP
fallback. If ARP is stale, configure the guest agent or a DHCP reservation rather
than hard-coding a changing address.
