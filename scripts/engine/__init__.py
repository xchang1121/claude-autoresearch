"""engine/ orchestration scripts invoked by hooks or Claude Code.

Blessed CLIs live here: baseline.py, pipeline.py, create_plan.py,
parse_args.py, and quick_check.py. Eval itself is not a standalone engine
script anymore; baseline.py and pipeline.py call task_config.run_eval, which
routes through utils.akg_eval and the formal KernelVerifier chain.
"""
