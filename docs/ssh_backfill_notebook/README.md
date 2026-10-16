# SSH backfill lab notebook (artifact generator)

Artifact (update in place, same URL): https://claude.ai/artifact/3KrSf6jB3zKo8Kedk1HTPv

```bash
python3 build.py <opym_local repo> <python with pytest+zmq+zarr> <path to bioimaging/scripts/storage_triage_notebook/notebook.tmpl.html>
# -> notebook.html  (publish to the same artifact URL)
```
The test count is computed by collecting `tests/test_stream_backfill.py`. Everything else comes from
`measurements.json`, which records each figure's provenance (the 2026-10-15 real-data run, the Globus rate in
CORE/docs/DATA_INGRESS.md, the PR diffstat). Re-measure and edit it when the facts change; update the "Status"
banner in the template when the service is deployed or the PC sender has run.
