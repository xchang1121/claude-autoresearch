"""Single source for paths that live OUTSIDE this scripts/ package.

Layout assumption — for claude-autoresearch the framework code and the
sibling content trees all sit under the repo root:

    <repo_root>/
      scripts/         <- this package (engine/, worker/, hooks/, utils/, ...)
      scripts/eval/    <- evaluation Python package (KernelVerifier +
                          adapters/{backend,dsl,framework}; migrated from
                          akg_agents/op/verifier — see scripts/eval/SPEC.md)
      skills/          <- per-DSL documentation tree (markdown only; no
                          executable verify/benchmark scripts live here)

Earlier revisions of autoresearch routed verify + benchmark through an
out-of-tree `skills/triton/kernel-verifier/scripts/` sibling. That CLI
contract is gone; the new eval entrypoint is the in-tree `scripts/eval`
package.

This module is the one place that encodes the layout — a tree move is
a one-line fix here instead of every consumer.
"""
import os

# external_paths.py → utils/ → scripts/ → repo root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

_SCRIPTS_ROOT = os.path.join(_REPO_ROOT, "scripts")


def eval_dir() -> str:
    """Dir holding the in-tree evaluation package (KernelVerifier,
    profiler, adapters/{backend,dsl,framework})."""
    return os.path.join(_SCRIPTS_ROOT, "eval")


def skills_dir() -> str:
    """Dir holding the per-DSL skill documentation tree."""
    return os.path.join(_REPO_ROOT, "skills")


def latency_refs_dir() -> str:
    """Back-compat alias for the skills tree root.

    Earlier revisions exposed a dedicated `skills/triton/latency-optimizer/
    references/` subtree of flat perf-tuning markdown. The new skills tree
    is DSL-partitioned (`skills/triton-ascend/`, `skills/triton-cuda/`,
    `skills/pypto/`, ...) and the references live inside each DSL's
    `fundamentals/` / `guides/` subdirs as SKILL.md files. The single
    root path the LLM Glob's against is the skills tree root itself.
    """
    return skills_dir()
