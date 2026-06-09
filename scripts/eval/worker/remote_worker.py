import httpx
import logging
import json
from typing import Tuple, Dict, Any, Callable, Optional

from .interface import (
    WorkerInterface,
    DEFAULT_EVAL_TIMEOUT_S,
    DEFAULT_GEN_REF_TIMEOUT_S,
)
from .manager import get_akg_env_var

logger = logging.getLogger(__name__)


# Connect / write / pool phase budgets and transient-retry count are read
# from the standard ``AKG_AGENTS_WORKER_*`` env var layer. Operators can
# tune for flaky tunnels via ``export AKG_AGENTS_WORKER_CONNECT_TIMEOUT_S=10``
# etc. without touching code. Defaults: connect short (5s) so dead ssh
# -L tunnels surface as ConnectError quickly; write/pool moderate;
# read timeout always comes from the per-call ``timeout`` arg because
# verify legitimately runs minutes on heavy DSLs.
def _connect_timeout_s() -> float:
    return float(get_akg_env_var("WORKER_CONNECT_TIMEOUT_S", "5.0"))


def _write_timeout_s() -> float:
    return float(get_akg_env_var("WORKER_WRITE_TIMEOUT_S", "30.0"))


def _pool_timeout_s() -> float:
    return float(get_akg_env_var("WORKER_POOL_TIMEOUT_S", "5.0"))


def _transient_retry_attempts() -> int:
    """Max attempts (incl. first) for the ConnectError retry path. Only
    consulted when ``on_transient_failure`` is wired. Default 2 = one
    retry after invoking the callback. Set to 1 for fail-fast."""
    return max(1, int(get_akg_env_var("WORKER_TRANSIENT_ATTEMPTS", "2")))


def _http_timeout(read_seconds: float) -> httpx.Timeout:
    """httpx.Timeout with connect/read/write/pool split so a hung daemon
    or dead tunnel can't make a single call swallow the full read budget."""
    return httpx.Timeout(
        connect=_connect_timeout_s(),
        read=read_seconds,
        write=_write_timeout_s(),
        pool=_pool_timeout_s(),
    )


class RemoteWorker(WorkerInterface):
    """
    Remote implementation of WorkerInterface.
    Delegates verification tasks to a remote VerificationService via HTTP.

    RemoteWorker 通过 HTTP API 管理远程服务器的设备池：
    - acquire_device(): 向远程服务器请求分配设备
    - release_device(): 归还设备给远程服务器
    - verify()/profile(): 发送任务到远程服务器执行

    ``on_transient_failure``: optional callback the worker invokes once
    after a ConnectError on long-running calls (verify/profile). Caller
    wires it to e.g. ``ar_cli worker --reconnect-tunnel`` so a dead ssh
    -L tunnel auto-heals between attempts.
    """
    def __init__(self, worker_url: str,
                 on_transient_failure: Optional[Callable[[], None]] = None):
        self.worker_url = worker_url.rstrip('/')
        self.on_transient_failure = on_transient_failure
    
    async def acquire_device(self, task_id: str = "unknown", timeout: float = None) -> int:
        """
        从远程服务器获取一个可用设备。
        
        Args:
            task_id: 任务ID（用于日志）
            timeout: 请求超时时间（秒）。默认为 None（无限等待），
                     因为在高并发 evolve 场景下，设备可能被长时间占用，
                     需要等待直到有设备可用。
                     取消等待：由上层 asyncio task 的 cancel 机制处理。
        
        Returns:
            int: 设备ID
        """
        acquire_url = f"{self.worker_url}/api/v1/acquire_device"
        
        try:
            # 使用 timeout=None (无限等待) 作为默认值，因为在并发高时
            # 设备可能被长时间占用，我们需要等待直到有设备可用。
            # httpx 默认 timeout 是 5s，之前硬编码是 10.0s，这在 evolve 流程中是不够的。
            async with httpx.AsyncClient(timeout=timeout) as client:
                data = {'task_id': task_id}
                response = await client.post(acquire_url, data=data)
                response.raise_for_status()
                
                result = response.json()
                device_id = result.get('device_id')
                logger.info(f"[{task_id}] Acquired remote device {device_id}")
                return device_id
        except Exception as e:
            logger.error(f"[{task_id}] Failed to acquire remote device: {e}")
            raise RuntimeError(f"Failed to acquire remote device: {e}")
    
    async def release_device(self, device_id: int, task_id: str = "unknown"):
        """
        归还设备给远程服务器。
        
        Args:
            device_id: 设备ID
            task_id: 任务ID（用于日志）
        """
        release_url = f"{self.worker_url}/api/v1/release_device"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                data = {'task_id': task_id, 'device_id': device_id}
                response = await client.post(release_url, data=data)
                response.raise_for_status()
                logger.info(f"[{task_id}] Released remote device {device_id}")
        except Exception as e:
            logger.error(f"[{task_id}] Failed to release remote device: {e}")

    async def get_doc(self, doc_name: str) -> str:
        """从远端 worker 拉取文档内容。"""
        doc_url = f"{self.worker_url}/api/v1/docs/{doc_name}"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(doc_url)
                response.raise_for_status()
                result = response.json()
                return result.get("content", "")
        except httpx.RequestError as e:
            logger.warning("Failed to fetch remote doc '%s' from %s: %s", doc_name, self.worker_url, e)
            return ""
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Remote worker returned %s for doc '%s': %s",
                e.response.status_code,
                doc_name,
                e.response.text,
            )
            return ""
        except Exception as e:
            logger.warning("Unexpected error when fetching remote doc '%s': %s", doc_name, e)
            return ""

    async def _post_with_reconnect(self, url: str, files, data,
                                   read_timeout: float, task_id: str):
        """POST helper: attempt up to ``_transient_retry_attempts()`` times;
        on each ConnectError invoke ``on_transient_failure`` (if set) and
        retry. Other HTTP errors and read timeouts bubble up to the
        caller so per-endpoint error handling can shape the response."""
        attempts = (_transient_retry_attempts()
                    if self.on_transient_failure is not None else 1)
        last_exc = None
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=_http_timeout(read_timeout)) as client:
                    response = await client.post(url, files=files, data=data)
                    response.raise_for_status()
                    return response.json()
            except httpx.ConnectError as e:
                last_exc = e
                if attempt + 1 < attempts:
                    logger.warning(
                        f"[{task_id}] 连接 worker {self.worker_url} 失败 "
                        f"（第 {attempt + 1}/{attempts} 次）；调用 "
                        f"on_transient_failure 后重试"
                    )
                    try:
                        self.on_transient_failure()
                    except Exception as cb_err:
                        logger.error(
                            f"[{task_id}] on_transient_failure 抛异常：{cb_err}"
                        )
                    continue
                raise
        raise last_exc

    async def verify(self, package_data: bytes, task_id: str, op_name: str, timeout: int = DEFAULT_EVAL_TIMEOUT_S) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Send verification task to remote worker.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log, artifacts)
        """
        verify_url = f"{self.worker_url}/api/v1/verify"

        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = {
                'task_id': task_id,
                'op_name': op_name,
                'timeout': str(timeout)
            }
            logger.info(f"[{task_id}] Sending verification request to {verify_url}")

            result = await self._post_with_reconnect(
                verify_url, files=files, data=data,
                read_timeout=timeout + 10, task_id=task_id,
            )
            success = result.get('success', False)
            log = result.get('log', '')
            artifacts = result.get('artifacts', {})

            if artifacts:
                logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")

            return success, log, artifacts

        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}. Please check if the worker service is running and accessible."
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Remote verification failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send profiling task to remote worker.

        Returns:
            Dict[str, Any]: 包含 gen_time, base_time, speedup, artifacts 等字段
        """
        profile_url = f"{self.worker_url}/api/v1/profile"
        timeout = profile_settings.get('timeout', DEFAULT_EVAL_TIMEOUT_S)
        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = {
                'task_id': task_id,
                'op_name': op_name,
                'profile_settings': json.dumps(profile_settings)
            }
            logger.info(f"[{task_id}] Sending profiling request to {profile_url}")

            result = await self._post_with_reconnect(
                profile_url, files=files, data=data,
                read_timeout=timeout + 10, task_id=task_id,
            )
            artifacts = result.get('artifacts', {})
            if artifacts:
                logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")

            return result

        except Exception as e:
            logger.error(f"[{task_id}] Remote profiling failed: {e}")
            return {'artifacts': {}}

    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str,
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send single task profiling request to remote worker.

        单独测量某段代码的执行性能，不进行 base vs generation 对比。

        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        profile_url = f"{self.worker_url}/api/v1/profile_single_task"
        timeout = profile_settings.get('timeout', DEFAULT_EVAL_TIMEOUT_S)
        try:
            files = {'package': ('package.tar', package_data, 'application/x-tar')}
            data = {
                'task_id': task_id,
                'op_name': op_name,
                'profile_settings': json.dumps(profile_settings)
            }
            logger.info(f"[{task_id}] Sending profile_single_task request to {profile_url}")

            result = await self._post_with_reconnect(
                profile_url, files=files, data=data,
                read_timeout=timeout + 10, task_id=task_id,
            )
            return result
                
        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}
        except Exception as e:
            error_msg = f"Remote profile_single_task failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}

    async def generate_reference(self, package_data: bytes, task_id: str, op_name: str, timeout: int = DEFAULT_GEN_REF_TIMEOUT_S) -> Tuple[bool, str, bytes]:
        """
        Send reference generation task to remote worker.
        
        用于 CUDA-to-Ascend 转换场景：在远程 GPU Worker 上执行 Triton-CUDA 代码，
        生成参考数据（.pt 文件）并返回其二进制内容。
        
        Args:
            package_data: 验证包数据（TAR bytes）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str, bytes]: (success, log, reference_data_bytes)
        """
        import base64
        
        generate_ref_url = f"{self.worker_url}/api/v1/generate_reference"
        
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                files = {'package': ('package.tar', package_data, 'application/x-tar')}
                data = {
                    'task_id': task_id,
                    'op_name': op_name,
                    'timeout': str(timeout)
                }
                
                logger.info(f"[{task_id}] Sending generate_reference request to {generate_ref_url}")
                
                response = await client.post(generate_ref_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                success = result.get('success', False)
                log = result.get('log', '')
                
                if success:
                    # reference_data 以 base64 编码传输
                    ref_data_b64 = result.get('reference_data', '')
                    if ref_data_b64:
                        ref_bytes = base64.b64decode(ref_data_b64)
                        logger.info(f"[{task_id}] Received reference data: {len(ref_bytes)} bytes")
                        return True, log, ref_bytes
                    else:
                        return False, f"No reference data in response:\n{log}", b''
                else:
                    return False, log, b''
                
        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except Exception as e:
            error_msg = f"Remote generate_reference failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
