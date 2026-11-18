# Experimental loader reference

This is the actual accepted experimental cache adapter, preserved for mechanism review. It is not installed by the package and is not a portable upstream implementation.

It expects a caller-provided frozen release directory with `config/code-manifest.json` and `config/weights-manifest.json`, exactly 48 streamed units, FP8-cast/Alpha=1/two-step=1/BF16, and the pinned LTX revision. `LTX_ALPHA_WEIGHT_CACHE=build|read` and `LTX_ALPHA_TRIM_FACTORS=1` select the paths. Cache reads rely on an outer release launcher that already verifies **all** asset SHA256 values and current file identities. That launcher and the derived weight bytes are not part of this public companion module. Do not call this reference with arbitrary unverified cache files.

Known integration debt: process-global monkeypatches, private scheduler members, specialized fingerprint, fixed unit count, asserts, and a zero-element no-factor sentinel. See the native-interface proposal in `../docs/UPSTREAM_PROPOSAL.md`. Do not use this reference as a security boundary or as a general-purpose cache API.

The exact-cache results document the tested original integration, not a promise about an arbitrary new integration. No cache blobs or base/adapter weights are redistributed.
