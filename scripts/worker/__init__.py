"""HTTP worker daemon for autoresearch.

`server.py` is a thin FastAPI wrapper that owns one device-pool queue
and exposes /api/v1/{status, run}. /run accepts a tar.gz package built
by `task_config.package_builder` and dispatches it through
`utils.eval_runner.local_eval` — the same code path direct local eval
uses, so local and remote transports cannot drift.
"""
