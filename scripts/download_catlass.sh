#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
#
# One-click install of the CATLASS C++ template library at the standard
# location <repo-root>/thirdparty/catlass. Required by the ascendc_catlass
# DSL (other DSLs ignore it). Pinned to a known-good commit so the
# repo's cmake snippets and any reference build artifacts stay coherent.
#
# Usage:
#   bash scripts/download_catlass.sh
#
# After this completes, eval/catlass_paths.resolve_catlass_root() will
# auto-discover the install — no CATLASS_ROOT env or config.yaml change
# needed for the standard deployment. See config.yaml `catlass:` section
# for the full resolution chain (and how to point at a custom install).
set -e

CATLASS_REPO_URL="${CATLASS_REPO_URL:-https://gitcode.com/cann/catlass.git}"
CATLASS_COMMIT="${CATLASS_COMMIT:-d60bf08c278c07d8fd1a74d3a4a4f590555d9ab9}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRDPARTY_DIR="${REPO_ROOT}/thirdparty"
TARGET_DIR="${THIRDPARTY_DIR}/catlass"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found on PATH" >&2
  exit 1
fi

mkdir -p "${THIRDPARTY_DIR}"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "catlass repo already at ${TARGET_DIR} — fetching latest refs"
  git -C "${TARGET_DIR}" fetch --tags origin
else
  if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${TARGET_DIR} exists and is non-empty but not a git repo — refusing to overwrite" >&2
    echo "       move it aside, then re-run." >&2
    exit 1
  fi
  echo "cloning ${CATLASS_REPO_URL} -> ${TARGET_DIR}"
  git clone "${CATLASS_REPO_URL}" "${TARGET_DIR}"
fi

if ! git -C "${TARGET_DIR}" rev-parse --verify "${CATLASS_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "fetching target commit ${CATLASS_COMMIT}"
  git -C "${TARGET_DIR}" fetch origin "${CATLASS_COMMIT}" || true
fi

if git -C "${TARGET_DIR}" rev-parse --verify "${CATLASS_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "checking out commit ${CATLASS_COMMIT}"
  git -C "${TARGET_DIR}" checkout "${CATLASS_COMMIT}"
else
  echo "WARNING: commit ${CATLASS_COMMIT} not reachable; staying on current ref" >&2
fi

if [ -d "${TARGET_DIR}/include/catlass" ]; then
  echo ""
  echo "catlass installed at ${TARGET_DIR}"
  echo "  include/catlass/ tree available — ascendc_catlass DSL ready to use"
else
  echo "ERROR: ${TARGET_DIR}/include/catlass missing after install" >&2
  exit 1
fi
