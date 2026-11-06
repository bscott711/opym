# ADR 0001: light base, heavy extras
Status: accepted (2026-11-06), run opym-source-of-truth decision D1.

## Context
opym is the source of truth for contracts other repos had copied by hand (wire protocol, bfill, layout-3 store,
timeline, registry, display rule). Those modules are almost all numpy/stdlib, but `import opym.<anything>` first ran
`opym/__init__.py`, which imported ipywidgets, matplotlib, scipy, skimage, tifffile, zarr, yaml, and
`opym/stream/__init__.py` also imported the zmq receiver. `import opym` failed in a base install (BF-12). ChimeraX,
pvpython, the rig PC and the viewers cannot take that stack.

## Decision
- `[project] dependencies = ["numpy"]` only. msgpack, pyzmq, zarr are not base.
- `opym/__init__.py` and `opym/stream/__init__.py` are lazy (module `__getattr__` over the old `__all__`), so every
  `from opym import X` and `from opym.stream import X` still works.
- `metadata.py` imports yaml inside `parse_mda_settings`; `lanes.py` imports fcntl inside `backfill_admission`;
  `stream/protocol.py` and `stream/bfill.py` import msgpack inside pack/unpack.

## Light module set (numpy only; enforced by `tests/test_light_base.py`, constant `LIGHT_MODULES`)
`opym`, `opym.stream`, `opym.stream.{protocol,bfill,bfill_run,backfill_runs}`, `opym.{timeline,registry,metadata,utils,
lanes,decon_config,ome_zarr_writer}`. Later tasks add `opym.stream.live_status`, `opym.display`, `opym.store`. The test
installs the checkout `--no-deps` into a numpy-only venv, imports each module and asserts no heavy package (scipy, skimage,
tifffile, zarr, yaml, msgpack, zmq, ipywidgets, matplotlib, ...) is in `sys.modules`. `pack_message`/`bfill.pack` need `stream`.

## Extras
- `stream`: msgpack, pyzmq, zarr (same bounds as before), numcodecs. `[stream]` alone does NOT make
  `opym.stream.receiver` importable (it reaches petakit and ipywidgets); the receiver needs `[server]`.
- `server`: all former base dependencies, plus scipy (imported in 4 files, was undeclared) and numcodecs.
- `viewer` and `dev`: unchanged. `all` = server + viewer. Python floor stays 3.10.

## Consumer install recipes (run at deploy only)
- Pin `opym @ git+https://github.com/bscott711/opym@<sha>`. Hatchling consumers declare `dependencies = ["opym"]` plus
  `[tool.uv.sources] opym = { git = ..., rev = <sha> }` (naparym-live pattern; no `allow-direct-references`).
- ChimeraX bundles and pvpython: `pip install --no-deps`. argus-desktop: a second unhashed `uv pip install --no-deps`.

## Consequences
- `import opym` no longer needs the viewer extra (BF-12). A missing heavy dependency now fails at the call site.
- bioimaging keeps the editable `../opym_local` path, declares `opym[server]` and relocks `uv.lock` in one PR (uv ignores
  an undefined extra, so it may merge first). Production pulls opym_local and bioimaging together, bioimaging no later
  than opym_local. Otherwise `uv run` re-syncs, rewrites the production `uv.lock`, and the next `uv sync` removes
  msgpack, ome-types, imagecodecs, scikit-learn (+ joblib, threadpoolctl, xsdata).
