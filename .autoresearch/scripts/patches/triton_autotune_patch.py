# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import inspect
import json
import os
import tempfile
from typing import Optional

# 全局变量存储配置信息
_collected_config_timings = {}

# ----------------------------------------------------------------------
# autotune disk cache — persist `Autotuner.cache` (the winner-config
# dict triton fills in by benchmarking) across process invocations.
#
# First run: triton benchmarks all configs, writes `self.cache[key] = best`.
#            After the call returns, we serialize self.cache to
#            `~/.autoresearch_cache/autotune/<fn_name>_<src_hash>.json`.
# Next run: before calling original_autotuner_run, we read the file and
#            populate `self.cache[key] = best` so the bench loop is
#            short-circuited by triton's own cache-hit fast path.
#
# Invalidation: src_hash is computed from the kernel's `triton.jit`
# source. Any kernel-source edit changes the hash → new cache file →
# old entries naturally stop being read. No manual eviction needed.
# ----------------------------------------------------------------------

_DISK_CACHE_LOADED_FOR: set = set()    # id(autotuner) → already loaded once
_DISK_CACHE_SAVED_FOR:  set = set()    # id(autotuner) → saw a save attempt (for logging dedup)


def _autotune_disk_cache_enabled() -> bool:
    try:
        from utils.settings import autotune_settings  # type: ignore
        return bool(autotune_settings().get("disk_cache_enabled", True))
    except Exception:
        return True


def _disk_cache_root() -> str:
    """Lives under `$HOME/.autoresearch_cache/autotune/` so eval-package
    tempdir wipes don't lose state, and concurrent eval processes on
    different devices share the same cache."""
    base = os.path.join(os.path.expanduser("~"), ".autoresearch_cache", "autotune")
    os.makedirs(base, exist_ok=True)
    return base


def _kernel_ident(autotuner) -> Optional[tuple]:
    """Return `(fn_name, src_hash_12)` keying the on-disk file, or None
    when the underlying JITFunction doesn't expose `__name__`/source."""
    fn = getattr(autotuner, 'base_fn', None) or getattr(autotuner, 'fn', None)
    if fn is None:
        return None
    name = getattr(fn, '__name__', None)
    src = getattr(fn, 'src', None) or ''
    if not src:
        inner = getattr(fn, 'fn', None)
        if inner is not None:
            src = getattr(inner, 'src', '') or ''
    if not name:
        return None
    try:
        src_bytes = src.encode("utf-8") if isinstance(src, str) else b''
    except Exception:
        src_bytes = b''
    h = hashlib.sha256(src_bytes).hexdigest()[:12]
    return name, h


def _disk_cache_path_for(autotuner) -> Optional[str]:
    ident = _kernel_ident(autotuner)
    if ident is None:
        return None
    name, h = ident
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    return os.path.join(_disk_cache_root(), f"{safe_name}_{h}.json")


_CONFIG_OPT_ATTRS = (
    "num_warps", "num_stages", "num_ctas", "num_buffers_warp_spec",
    "num_consumer_groups", "reg_dec_producer", "reg_inc_consumer",
    "maxnreg",
)


def _serialize_config(config) -> Optional[dict]:
    """Capture enough of a `triton.Config` to reconstruct later."""
    if config is None:
        return None
    try:
        out: dict = {"kwargs": dict(getattr(config, "kwargs", {}) or {})}
    except Exception:
        return None
    for attr in _CONFIG_OPT_ATTRS:
        v = getattr(config, attr, None)
        if v is not None:
            try:
                json.dumps(v)        # ensure JSON-serialisable
                out[attr] = v
            except TypeError:
                continue
    return out


def _deserialize_config(d: dict):
    """Reconstruct a `triton.Config` from a serialized dict. Returns None
    when triton is unavailable or the dict shape doesn't match the
    installed triton version's Config signature."""
    if not d or not _TRITON_AVAILABLE:
        return None
    try:
        kwargs = dict(d.get("kwargs") or {})
        sig_params = set(inspect.signature(triton.Config.__init__).parameters)
        extra = {k: d[k] for k in d if k != "kwargs" and k in sig_params}
        return triton.Config(kwargs, **extra)
    except Exception:
        return None


def _key_to_str(key) -> str:
    try:
        return json.dumps(list(key), default=str)
    except Exception:
        return json.dumps([str(x) for x in (key if isinstance(key, tuple) else (key,))])


def _str_to_key(s: str):
    try:
        return tuple(json.loads(s))
    except Exception:
        return None


def _maybe_load_disk_cache(autotuner) -> None:
    """First call per Autotuner instance: load disk cache, merge entries
    into `self.cache`. Idempotent across the process lifetime; later
    calls return immediately.
    """
    if id(autotuner) in _DISK_CACHE_LOADED_FOR:
        return
    _DISK_CACHE_LOADED_FOR.add(id(autotuner))

    if not _autotune_disk_cache_enabled():
        return

    path = _disk_cache_path_for(autotuner)
    if path is None or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    entries = data.get("entries") or {}

    cache = getattr(autotuner, "cache", None)
    if cache is None:
        try:
            autotuner.cache = {}
            cache = autotuner.cache
        except Exception:
            return

    loaded = 0
    for key_str, cfg_dict in entries.items():
        key = _str_to_key(key_str)
        if key is None:
            continue
        cfg = _deserialize_config(cfg_dict)
        if cfg is None:
            continue
        try:
            cache[key] = cfg
            loaded += 1
        except Exception:
            continue
    if loaded and os.getenv("AR_AUTOTUNE_DEBUG"):
        print(f"[autotune] loaded {loaded} cache entries from {path}")


def _save_disk_cache(autotuner) -> None:
    """After `original_autotuner_run` returns: snapshot `self.cache` to
    disk so the next process can skip the bench loop. Atomic write
    (tmp + replace) to avoid torn JSON when two devices race.
    """
    if not _autotune_disk_cache_enabled():
        return
    cache = getattr(autotuner, "cache", None)
    if not cache:
        return
    path = _disk_cache_path_for(autotuner)
    if path is None:
        return

    ident = _kernel_ident(autotuner)
    fn_name, src_hash = ident if ident else ("unknown", "0")

    entries: dict = {}
    for key, cfg in cache.items():
        cfg_dict = _serialize_config(cfg)
        if cfg_dict is None:
            continue
        entries[_key_to_str(key)] = cfg_dict
    if not entries:
        return

    payload = {"fn_name": fn_name, "src_hash": src_hash, "entries": entries}
    try:
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".at_", suffix=".json.tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as e:
        if os.getenv("AR_AUTOTUNE_DEBUG"):
            print(f"[autotune] disk-cache save failed for {path}: {e}")
        return
    if id(autotuner) not in _DISK_CACHE_SAVED_FOR:
        _DISK_CACHE_SAVED_FOR.add(id(autotuner))
        if os.getenv("AR_AUTOTUNE_DEBUG"):
            print(f"[autotune] saved {len(entries)} cache entries to {path}")

# ============================================================================
# AR_restore_copy Triton kernel
# 参考 l2_cache_clear.py 的设计：使用专用 kernel，
# 便于在 profiler 的 op_statistic.csv 中按名字精确过滤。
# ============================================================================

AR_RESTORE_COPY_KERNEL_NAME = "AR_restore_copy"

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

if _TRITON_AVAILABLE:
    @triton.jit
    def AR_restore_copy(
        dst_ptr, src_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
    ):
        """
        restore_value 专用 copy kernel。

        kernel 名称在 profiler 中显示为 AR_restore_copy，
        可精确过滤，不会误删用户代码中的 TensorMove 等同名操作。
        """
        pid = tl.program_id(0)
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
        for block_idx in range(pid, num_blocks, CORE_NUM):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            data = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, data, mask=mask)


def _get_vec_core_num():
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except Exception:
        return 40


def ar_restore_copy(dst, src):
    """用 AR_restore_copy kernel 执行 tensor copy，替代 tensor.copy_()。"""
    import torch
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num = _get_vec_core_num()
    BLOCK_SIZE = 1024
    grid = (core_num,)
    AR_restore_copy[grid](dst_flat, src_flat, n,
                           BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


def _restore_saved_tensors(saved, args):
    """Restore saved output tensors back to the live kernel arguments."""
    for idx, saved_val in saved.items():
        ar_restore_copy(args[idx], saved_val)


def _wrap_kernel_call_with_restore(kernel_call, restore_info):
    """Wrap benchmark calls with Triton-like pre/post restore semantics."""
    if restore_info is None:
        return kernel_call

    saved = restore_info['saved']
    args = restore_info['args']

    def wrapped_call():
        _restore_saved_tensors(saved, args)
        try:
            return kernel_call()
        finally:
            # Leave every benchmark iteration with the original output state
            # so a later config cannot inherit stale values from an earlier one.
            _restore_saved_tensors(saved, args)

    return wrapped_call


# ============================================================================
# _bench patch: 禁用原生 restore_value 的 copy_()，
# 让 kernel_call 只包含纯 kernel，restore 交给 benchmarker 用命名 kernel 做。
# ============================================================================

_restore_info = None


def _patch_autotuner_bench(autotuner_module):
    """Patch Autotuner._bench，在 restore_value 场景下接管 pre_hook。"""
    original_bench = getattr(autotuner_module.Autotuner, '_bench', None)
    if original_bench is None:
        return
    if getattr(original_bench, '_ar_bench_patched', False):
        return

    _noop = lambda *a, **kw: None

    def patched_bench(self, *args, config, **meta):
        global _restore_info

        if not (_TRITON_AVAILABLE and hasattr(self, 'restore_value') and self.restore_value):
            _restore_info = None
            return original_bench(self, *args, config=config, **meta)

        saved = {}
        for name in self.restore_value:
            idx = self.fn.arg_names.index(name)
            saved[idx] = args[idx].clone()
        _restore_info = {'saved': saved, 'args': list(args)}

        orig_rv = self.restore_value
        orig_ph = getattr(self, 'pre_hook', None)
        orig_posth = getattr(self, 'post_hook', None)
        self.restore_value = None
        self.pre_hook = _noop
        self.post_hook = _noop

        try:
            result = original_bench(self, *args, config=config, **meta)
        finally:
            self.restore_value = orig_rv
            self.pre_hook = orig_ph
            self.post_hook = orig_posth
            _restore_info = None

        return result

    patched_bench._ar_bench_patched = True
    autotuner_module.Autotuner._bench = patched_bench


# ============================================================================
# 需要过滤的底层实现参数
# ============================================================================

_FILTERED_CONFIG_PARAMS = {
    'num_warps',
    'num_ctas',
    'num_stages',
    'num_buffers_warp_spec',
    'num_consumer_groups',
    'reg_dec_producer',
    'reg_inc_consumer',
    'maxnreg'
}


def _filter_config_string(config_str: str) -> str:
    """过滤配置字符串，移除底层实现参数"""
    params = []
    for param in config_str.split(','):
        param = param.strip()
        if not param:
            continue
        if ':' in param:
            param_name = param.split(':', 1)[0].strip()
        elif '=' in param:
            param_name = param.split('=', 1)[0].strip()
        else:
            params.append(param)
            continue
        if param_name not in _FILTERED_CONFIG_PARAMS:
            params.append(param)
    return ', '.join(params)


def patch_triton_autotuner():
    """动态补丁 triton autotuner，添加配置信息收集 + _bench restore_value 接管。"""
    try:
        import triton.runtime.autotuner as autotuner_module
    except ImportError:
        return True

    try:
        import triton.runtime.autotiling_tuner as autotiling_module
    except ImportError:
        autotiling_module = None

    if not hasattr(autotuner_module, 'Autotuner'):
        return True

    original_autotuner_run = getattr(autotuner_module.Autotuner, 'run', None)
    if original_autotuner_run is None:
        return True
    if getattr(original_autotuner_run, '_ar_run_patched', False):
        return True

    original_autotiling_run = None
    if autotiling_module and hasattr(autotiling_module, 'AutoTilingTuner'):
        original_autotiling_run = getattr(autotiling_module.AutoTilingTuner, 'run', None)

    # Patch _bench 接管 restore_value
    _patch_autotuner_bench(autotuner_module)

    def _process_config_timings(self):
        if not (hasattr(self, 'best_config') and
                hasattr(self, 'configs_timings') and
                self.configs_timings and
                isinstance(self.configs_timings, dict)):
            return

        func_name = "unknown_function"
        try:
            if hasattr(self, 'base_fn') and hasattr(self.base_fn, '__name__'):
                func_name = self.base_fn.__name__
            elif hasattr(self, 'fn') and hasattr(self.fn, '__name__'):
                func_name = self.fn.__name__
        except (AttributeError, TypeError):
            pass

        try:
            sorted_timings = sorted(self.configs_timings.items(), key=lambda x: x[1])
            config_data = []
            for i, (config, timing) in enumerate(sorted_timings):
                try:
                    is_best = config == self.best_config
                    timing_value = timing[0] if isinstance(timing, list) else timing
                    timing_us = timing_value
                    config_str = _filter_config_string(str(config))
                    config_data.append({
                        "config": config_str,
                        "timing_us": float(timing_us),
                        "is_best": is_best,
                        "rank": i + 1
                    })
                except (TypeError, ValueError, AttributeError):
                    continue

            if config_data:
                global _collected_config_timings
                if func_name not in _collected_config_timings:
                    _collected_config_timings[func_name] = config_data

                    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                        print(f"All config timings for {func_name}:")
                        for i, (config, timing) in enumerate(sorted_timings):
                            try:
                                status = " (BEST)" if config == self.best_config else ""
                                timing_value = timing[0] if isinstance(timing, list) else timing
                                timing_us = timing_value
                                config_str = _filter_config_string(str(config))
                                print(f"  Config {i+1}: {config_str} -> {timing_us:.4f}us{status}")
                            except (TypeError, ValueError, AttributeError):
                                continue

        except (TypeError, ValueError, AttributeError):
            pass

    def patched_autotuner_run(self, *args, **kwargs):
        # Pre-warm `self.cache` from disk on first call for this instance
        # so triton's built-in cache-hit fast path skips the bench loop
        # entirely. No-op when the disk file doesn't exist (cold first
        # round) or `autotune.disk_cache_enabled` is false in config.
        try:
            _maybe_load_disk_cache(self)
        except Exception:
            pass
        result = original_autotuner_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        # Save AFTER the bench so any newly-picked winners land in the
        # next process's pre-warm. Idempotent (atomic replace), cheap
        # to call on every run.
        try:
            _save_disk_cache(self)
        except Exception:
            pass
        return result

    def patched_autotiling_run(self, *args, **kwargs):
        try:
            _maybe_load_disk_cache(self)
        except Exception:
            pass
        result = original_autotiling_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        try:
            _save_disk_cache(self)
        except Exception:
            pass
        return result

    try:
        patched_autotuner_run._ar_run_patched = True
        autotuner_module.Autotuner.run = patched_autotuner_run
    except (AttributeError, TypeError):
        pass

    if original_autotiling_run is not None:
        try:
            patched_autotiling_run._ar_run_patched = True
            autotiling_module.AutoTilingTuner.run = patched_autotiling_run
        except (AttributeError, TypeError):
            pass

    return True


def get_collected_config_timings():
    global _collected_config_timings
    return _collected_config_timings.copy()


def clear_collected_config_timings():
    global _collected_config_timings
    _collected_config_timings = {}


def _autotune_bench_settings() -> dict:
    """Pull autotune-bench knobs from `utils.settings.autotune_settings()`,
    falling back to lightweight defaults when settings aren't reachable
    (e.g. eval running in the tar-extracted worker tempdir where the
    scripts root isn't on sys.path)."""
    try:
        from utils.settings import autotune_settings  # type: ignore
        s = autotune_settings()
        return {
            "warmup":        int(s.get("benchmark_warmup", 1)),
            "active":        int(s.get("benchmark_active", 3)),
            "clear_l2":      bool(s.get("benchmark_clear_l2", False)),
        }
    except Exception:
        return {"warmup": 1, "active": 3, "clear_l2": False}


def patch_driver_benchmarker():
    """补丁 driver.active.get_benchmarker()，让 autotune 使用 profiler_npu。

    autotune-time benchmark settings (`warmup` / `active` / `clear_l2`)
    come from `autotune.benchmark_*` in .autoresearch/config.yaml. They
    are intentionally LIGHTER than the user-facing measurement — autotune
    only needs to relative-order configs, not measure final latency.
    Historical defaults were warmup=5 / active=30 + per-iter L2 clear,
    which made autotune dominate wallclock on multi-shape kernels
    (51 cases × 4 configs × 30 launches × 2 = 12k profiled launches
    just to pick winners). config.yaml ships warmup=1 / active=3 /
    no L2-clear, ~12× lighter at autotune time.

    当 _restore_info 不为空时（即 _bench 禁用了原生 restore_value），
    benchmarker 自动用 AR_restore_copy kernel 包装 kernel_call，
    profiler 按 kernel 名字精确过滤，不会误删用户的 TensorMove 操作。
    """
    try:
        from triton.runtime import driver

        if hasattr(driver.active.get_benchmarker, '_ar_patched'):
            return True

        original_get_benchmarker = driver.active.get_benchmarker
        bench_cfg = _autotune_bench_settings()

        def patched_get_benchmarker():
            def custom_benchmarker(kernel_call, quantiles=(0.5, 0.2, 0.8)):
                fn_to_profile = _wrap_kernel_call_with_restore(kernel_call, _restore_info)

                try:
                    from verifier.profiler import profiler_npu

                    time_us = profiler_npu(
                        fn_to_profile,
                        warmup=bench_cfg["warmup"],
                        active=bench_cfg["active"],
                        suppress_warnings=True,
                        clear_l2_cache=bench_cfg["clear_l2"],
                        l2_clear_kernel_name="AR_l2cache_clear",
                        filter_restore_copy=(_restore_info is not None),
                    )
                    return [time_us] * 3

                except ImportError:
                    original_benchmarker = original_get_benchmarker()
                    return original_benchmarker(fn_to_profile, quantiles)

            return custom_benchmarker

        driver.active.get_benchmarker = patched_get_benchmarker
        driver.active.get_benchmarker._ar_patched = True
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: Failed to patch driver benchmarker: {e}")
        return False


def apply_triton_patches():
    """应用所有triton补丁"""
    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()
    return success1 or success2


if __name__ != "__main__":
    apply_triton_patches()

if __name__ == "__main__":
    print("Testing Triton patches...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()

    if success1:
        print("Autotuner patch applied successfully!")
    if success2:
        print("Driver benchmarker patch applied successfully!")

    if not any([success1, success2]):
        print("Failed to apply patches")
