"""
GPU Memory Monitor for Windows with shared memory support.
Combines NVML for dedicated GPU memory and PDH for shared GPU memory.
"""

import sys
import time
import warnings

# ---------- 依赖检查 ----------
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

if sys.platform == 'win32':
    try:
        import win32pdh
        HAS_PDH = True
    except ImportError:
        HAS_PDH = False
else:
    HAS_PDH = False

_nvml_init_done = False

def _query_pdh_counter(instance_name, counter_name, fmt=win32pdh.PDH_FMT_DOUBLE):
    """临时查询 PDH 计数器值（返回字节数，失败返回 0.0）"""
    if not HAS_PDH:
        return 0.0
    q = win32pdh.OpenQuery()
    try:
        path = f"\\GPU Adapter Memory({instance_name})\\{counter_name}"
        h = win32pdh.AddCounter(q, path)
        # PDH 标准操作：采集两次取最新值
        win32pdh.CollectQueryData(q)
        time.sleep(0.05)
        win32pdh.CollectQueryData(q)
        _, val = win32pdh.GetFormattedCounterValue(h, fmt)
        return float(val)
    except Exception:
        return 0.0
    finally:
        try: win32pdh.CloseQuery(q)
        except: pass


class GPUMemoryMonitor:
    def __init__(self, gpu_index=0, adapter_name=None):
        self.gpu_index = gpu_index
        self.nvml_handle = None
        self.pdh_query = None
        self.pdh_counter_handle = None
        self.shared_adapter_name = adapter_name
        self._closed = False

        global _nvml_init_done
        # ---- NVML 初始化 ----
        if HAS_NVML:
            try:
                if not _nvml_init_done:
                    pynvml.nvmlInit()
                    _nvml_init_done = True
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                warnings.warn(f"NVML 初始化失败 (GPU {gpu_index}): {e}")
                self.nvml_handle = None

        # ---- PDH 初始化 ----
        if sys.platform == 'win32' and HAS_PDH:
            try:
                self._init_pdh()
            except Exception as e:
                warnings.warn(f"PDH 初始化失败: {e}")

    def _init_pdh(self):
        self.pdh_query = win32pdh.OpenQuery()
        try:
            _, instances = win32pdh.EnumObjectItems(None, None, "GPU Adapter Memory", win32pdh.PERF_DETAIL_WIZARD)
        except Exception as e:
            raise RuntimeError(f"无法枚举 PDH GPU 实例: {e}")

        if not instances:
            raise RuntimeError("系统中未找到 GPU Adapter Memory 实例。")

        if self.shared_adapter_name is None and self.nvml_handle is not None:
            # 🎯 核心匹配逻辑：对比 NVML 与 PDH 的“已用专用显存”
            try:
                nvml_mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                target_used = nvml_mem.used  # bytes
            except:
                target_used = 0

            if target_used > 10 * 1024**2:  # > 10MB 才认为有效
                best_inst = None
                best_diff = float('inf')
                for inst in instances:
                    pdh_used = _query_pdh_counter(inst, "Dedicated Usage")
                    if pdh_used > 0:
                        diff = abs(pdh_used - target_used)
                        if diff < best_diff:
                            best_diff = diff
                            best_inst = inst

                # 允许 25% 相对误差 或 1.5GB 绝对误差（应对瞬时波动/驱动差异）
                if best_inst and (best_diff < target_used * 0.25 or best_diff < 1.5 * 1024**3):
                    self.shared_adapter_name = best_inst
                else:
                    # 回退：选专用显存占用最高的实例（独显概率极大）
                    best_inst = max(instances, key=lambda i: _query_pdh_counter(i, "Dedicated Usage"))
                    self.shared_adapter_name = best_inst
                    warnings.warn(f"显存指纹匹配偏差较大，已自动 fallback 到专用显存占用最高的实例: {self.shared_adapter_name}")
            else:
                # NVML 显存极低时，直接选 Dedicated Usage 最大的
                best_inst = max(instances, key=lambda i: _query_pdh_counter(i, "Dedicated Usage"))
                self.shared_adapter_name = best_inst

        elif self.shared_adapter_name is None:
            # 无 NVML 时的兜底
            best_inst = max(instances, key=lambda i: _query_pdh_counter(i, "Dedicated Usage"))
            self.shared_adapter_name = best_inst
            warnings.warn(f"NVML 不可用，已自动选择专用显存占用最高的实例: {self.shared_adapter_name}")

        # 绑定 Shared Usage 计数器
        counter_path = f"\\GPU Adapter Memory({self.shared_adapter_name})\\Shared Usage"
        try:
            self.pdh_counter_handle = win32pdh.AddCounter(self.pdh_query, counter_path)
            win32pdh.CollectQueryData(self.pdh_query)
            time.sleep(0.01)
        except Exception as e:
            raise RuntimeError(f"添加 PDH 计数器失败 {counter_path}: {e}")

    def get_dedicated_memory_mb(self):
        if self.nvml_handle is None: return 0.0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            return info.used / (1024 ** 2)
        except: return 0.0

    def get_shared_memory_mb(self):
        if self.pdh_counter_handle is None: return 0.0
        try:
            win32pdh.CollectQueryData(self.pdh_query)
            _, val = win32pdh.GetFormattedCounterValue(self.pdh_counter_handle, win32pdh.PDH_FMT_DOUBLE)
            return val / (1024 ** 2)
        except: return 0.0

    def get_total_memory_mb(self):
        if self.nvml_handle is None: return 0.0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            return info.total / (1024 ** 2)
        except: return 0.0

    def get_usage(self):
        used = self.get_dedicated_memory_mb() + self.get_shared_memory_mb()
        return {
            'dedicated_gpu_memory_mb': self.get_dedicated_memory_mb(),
            'shared_gpu_memory_mb': self.get_shared_memory_mb(),
            'total_gpu_memory_mb': self.get_total_memory_mb(),
            'total_used_memory_mb': used
        }

    def close(self):
        if self._closed: return
        self._closed = True
        if self.pdh_query is not None:
            try: win32pdh.CloseQuery(self.pdh_query)
            except: pass
        global _nvml_init_done
        if HAS_NVML and _nvml_init_done:
            try:
                pynvml.nvmlShutdown()
                _nvml_init_done = False
            except: pass

    def __del__(self):
        self.close()


# ---------- 全局单例接口 ----------
_default_monitor = None

def init(gpu_index=0, adapter_name=None):
    global _default_monitor
    if _default_monitor is not None:
        _default_monitor.close()
    _default_monitor = GPUMemoryMonitor(gpu_index, adapter_name)
    return _default_monitor

def get_gpu_memory_usage():
    global _default_monitor
    if _default_monitor is None: init()
    return _default_monitor.get_usage()['total_used_memory_mb']

def get_gpu_memory_details():
    global _default_monitor
    if _default_monitor is None: init()
    return _default_monitor.get_usage()

def close():
    global _default_monitor
    if _default_monitor is not None:
        _default_monitor.close()
        _default_monitor = None

def debug_list_instances():
    """调试用：列出当前系统所有 GPU PDH 实例及它们的专用显存占用"""
    if not HAS_PDH: return []
    try:
        _, instances = win32pdh.EnumObjectItems(None, None, "GPU Adapter Memory", win32pdh.PERF_DETAIL_WIZARD)
        res = []
        for inst in instances:
            used_mb = _query_pdh_counter(inst, "Dedicated Usage") / (1024**2)
            res.append((inst, f"{used_mb:.1f} MB"))
        return res
    except: return []

# 导入时静默初始化（默认 NVML index 0 即 RTX 3090 Ti）
init()