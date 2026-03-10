# MLSmartSampler 实现计划

## 概述

在 `smart_sampler` 模块中新增 `MLSmartSampler` 类，采用三层漏斗型架构（硬过滤 → 快速触发 → 精细验证），接口与现有 `SmartSampler` 完全一致，可在 `LivePipeline` 中自由切换。

## 用户选择

- **imagehash**: 允许新增 `imagehash` 依赖
- **LivePipeline 切换**: 仅更新 `__init__.py` 导出，不修改 `LivePipeline`
- **光流策略**: 使用 Lucas-Kanade 稀疏光流（约50个特征点）

---

## 新增文件清单

| 文件 | 用途 |
|------|------|
| `src/python_test/smart_sampler/hard_filter.py` | Layer 0: 硬性过滤器 |
| `src/python_test/smart_sampler/fast_triggers.py` | Layer 1: 快速触发器组 |
| `src/python_test/smart_sampler/frame_validator.py` | Layer 2: 精细验证器 |
| `src/python_test/smart_sampler/ml_smart_sampler.py` | 主控类（对外接口） |

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/python_test/smart_sampler/__init__.py` | 新增 `MLSmartSampler` 导出 |

---

## Step 1: hard_filter.py — Layer 0 硬性过滤器

```python
class HardFilter:
    def __init__(self,
                 min_frame_interval: float = 1.0,
                 still_threshold: float = 2.0,
                 black_threshold: int = 10,
                 white_threshold: int = 245,
                 entropy_low: float = 3.0,
                 entropy_high: float = 7.0,
                 size_change_tolerance: float = 0.05) -> None

    def check(self, frame: np.ndarray, timestamp: float) -> tuple[bool, str]:
        """返回 (通过, 拒绝原因)。按顺序执行：
        1. 时间间隔控制 — ts - last_accept_ts < min_frame_interval → 拒绝
        2. 静止帧检测 — cv2.absdiff(当前灰度小图, 上一帧灰度小图), mean(diff) < still_threshold → 拒绝
        3. 黑屏/白屏检测 — mean(gray) < 10 或 > 245 → 拒绝
        4. 熵异常检测 — calcHist → entropy < 3.0 或 > 7.0 → 拒绝
        5. 分辨率一致性 — 像素面积变化 > 5% → 拒绝
        """

    def reset(self) -> None
    @property
    def reject_count(self) -> int
```

**实现要点**:
- 灰度图下采样到 160×90 后计算，确保 <0.1ms
- 缓存上一帧灰度小图 `_prev_gray_small`
- 通过时更新 `_last_accept_ts`

---

## Step 2: fast_triggers.py — Layer 1 快速触发器

```python
class FastTriggers:
    def __init__(self,
                 motion_method: str = 'MOG2',
                 motion_threshold: float = 0.3,
                 scene_switch_threshold: float = 0.5,
                 anomaly_zscore_threshold: float = 2.0,
                 periodic_interval: float = 30.0) -> None

    def detect(self, frame: np.ndarray, timestamp: float) -> list[str]:
        """OR 逻辑，返回触发器名称列表（空 = 无触发）"""

    def reset(self) -> None
    @property
    def trigger_count(self) -> int
    @property
    def motion_method(self) -> str
    @property
    def last_motion_score(self) -> float
    @property
    def last_fg_mask(self) -> np.ndarray | None
    @property
    def last_diff_mask(self) -> np.ndarray | None  # 帧差掩码，供scene_switch/anomaly裁剪用
```

### 四个触发器（内部方法）

**触发器1: 剧烈运动 `_check_motion()`**
- 下采样到 320×180
- 帧差法 `cv2.absdiff` + 阈值分割
- 连通域分析 `cv2.connectedComponentsWithStats`
- 复用 `MotionDetector`（背景减除）获取 fg_mask 和 motion_score
- 触发条件: motion_score > motion_threshold

**触发器2: 场景切换 `_check_scene_switch()`**
- HSV 直方图比对: `cv2.calcHist` + `cv2.compareHist(HISTCMP_CORREL)`
- 分块直方图（3×3 九宫格），逐块比对取最小相关系数
- 感知哈希预筛: `imagehash.phash()`，汉明距离 > 15 触发
- 触发条件: 相关系数 < scene_switch_threshold

**触发器3: 异常事件 `_check_anomaly()`**
- 维护滑动窗口（最近30帧特征）: [motion_intensity, color_variance, edge_density]
- Z-Score 计算: `z = (current - mean) / (std + eps)`
- 触发条件: 任一维度 |z| > anomaly_zscore_threshold

**触发器4: 周期强制 `_check_periodic()`**
- `timestamp - last_periodic_ts >= periodic_interval`
- 触发后更新 `last_periodic_ts`

---

## Step 3: frame_validator.py — Layer 2 精细验证器

```python
class FrameValidator:
    def __init__(self,
                 ssim_threshold: float = 0.80,
                 score_window_size: int = 10,
                 peak_ratio: float = 1.10,
                 nms_interval: float = 0.5) -> None

    def validate(self, frame: np.ndarray, timestamp: float,
                 triggers: list[str]) -> dict[str, Any]:
        """返回 {passed, score, features, ssim_score, is_peak, window_mean}"""

    def update_reference(self, frame: np.ndarray) -> None
    def reset(self) -> None
    @property
    def pass_count(self) -> int
    @property
    def ssim_threshold(self) -> float
```

### 五维特征提取

| 特征 | 方法 | 归一化到 0~1 |
|------|------|-------------|
| motion_persistence | `cv2.calcOpticalFlowPyrLK`，约50个特征点，统计位移>阈值的比例 | 比例本身 |
| structural_change | Sobel 边缘 + Edge IoU: `intersection/union` | 1 - IoU |
| content_semantics | `imagehash.phash` 汉明距离 | distance / 64 |
| temporal_coherence | 与最近5个关键帧的平均 SSIM | 1 - avg_ssim |
| anomaly_significance | 从触发器列表推断: 有anomaly=0.8, 多触发=0.5, 其他=0.2 | 直接赋值 |

### 动态权重

| 触发类型 | 权重调整 |
|----------|----------|
| 纯运动触发 ('motion' only) | motion×1.5, structural×0.5 |
| 纯场景触发 ('scene_switch' only) | structural×1.3, content×1.3, motion×0.7 |
| 多触发并存 | 保持基础权重，综合阈值降低10% |
| 周期强制 ('periodic' only) | 权重均等，验证阈值提高 |

### 时序平滑 & 峰值检测 & NMS

- 滑动窗口维护最近 `score_window_size` 个分数
- 峰值检测: `score > peak_ratio * window_mean` 且为窗口内局部极大值
- NMS: `timestamp - last_pass_ts < nms_interval` 时比较分数，保留更高分
- 周期强制触发跳过峰值检测直接通过

---

## Step 4: ml_smart_sampler.py — 主控类

```python
class MLSmartSampler:
    """分层漏斗型智能帧过滤器 - 接口与 SmartSampler 完全一致"""

    def __init__(self,
                 enable_smart_sampling: bool = True,
                 motion_method: str = 'MOG2',
                 ssim_threshold: float = 0.80,
                 motion_threshold: float = 0.3,
                 backup_interval: float = 30.0,
                 min_frame_interval: float = 1.0) -> None

    @property
    def frame_count(self) -> int
    @property
    def enable_smart(self) -> bool

    def sample(self, frames: Iterator[tuple[np.ndarray, float]]) -> Iterator[Dict[str, Any]]
    def reset(self) -> None
    def get_statistics(self) -> Dict[str, Any]
```

### 参数映射

| 构造函数参数 | 内部去向 |
|-------------|---------|
| `enable_smart_sampling` | `self.enable_smart` 总开关 |
| `motion_method` | → `FastTriggers(motion_method=...)` |
| `ssim_threshold` | → `FrameValidator(ssim_threshold=...)` |
| `motion_threshold` | → `FastTriggers(motion_threshold=...)` |
| `backup_interval` | → `FastTriggers(periodic_interval=...)` |
| `min_frame_interval` | → `HardFilter(min_frame_interval=...)` |

### sample() 核心流程

```
for frame_np, ts in frames:
    frame_idx = self._input_frame_count   # 全局帧序号（含被拒帧）
    self._input_frame_count += 1

    if not enable_smart:  # 降级为纯时间采样
        if time_based: yield {..., 'source': 'time'}; continue

    # Layer 0: 硬过滤
    passed, reason = self.hard_filter.check(frame_np, ts)
    if not passed:
        logger.debug("Layer0拒绝 帧#%d 原因=%s", frame_idx, reason)
        continue

    # Layer 1: 快速触发
    triggers = self.fast_triggers.detect(frame_np, ts)
    if not triggers:
        logger.debug("Layer1无触发 帧#%d", frame_idx)
        continue

    # Layer 2: 精细验证
    result = self.frame_validator.validate(frame_np, ts, triggers)
    if not result['passed']:
        logger.debug("Layer2拒绝 帧#%d 分数=%.3f", frame_idx, result['score'])
        continue

    # 通过！语义变化区域裁剪 + 组装返回字典
    source = _determine_source(triggers)  # 运动 > 变化(scene_switch) > 异常(anomaly) > 时间(periodic)

    # ── 语义区域裁剪（节省VLM token）──
    # 策略：根据触发类型选择最佳变化掩码，复用 ForegroundExtractor 裁剪
    #
    # 1. motion 触发 → 直接使用 fast_triggers 缓存的 fg_mask（背景减除掩码）
    # 2. scene_switch 触发 → 用帧差法生成差异掩码：
    #    diff = cv2.absdiff(current_gray, prev_gray) → 二值化 → 作为 mask
    # 3. anomaly 触发 → 同 scene_switch，使用帧差掩码
    # 4. periodic 触发（保底）→ 不裁剪，输出整帧
    #
    # 所有掩码统一送入 ForegroundExtractor.extract_foreground(frame, mask)
    # 得到 cropped_frame + bbox，再计算 compression_ratio

    # 构建标准返回字典 + 日志 [送VLM]
    yield result_dict
```

### 返回字典格式（与 SmartSampler 完全一致）

```python
{
    'image': PIL.Image,            # BGR→RGB PIL图像（裁剪后）
    'timestamp': float,
    'frame_index': int,            # 全局输入帧序号
    'significant': bool,           # 非周期触发 = True
    'source': str,                 # 'smart'/'time'
    'original_frame': np.ndarray,
    'cropped_frame': np.ndarray,
    'bbox': (x, y, w, h),
    'compression_ratio': float,
    'change_metrics': {
        'ssim_score': float,       # 来自 validator
        'combined_score': float,   # validator fusion score
        'motion_score': float      # 来自 fast_triggers
    }
}
```

### 日志格式（匹配现有规范）

```
[送VLM] 帧#N | ts=X.XXXs | 相似度=X.XXX(阈值<X.XX) | 运动=是/否(得分=X.XXX) | 综合=X.XXX | 来源=运动/变化/时间 | 裁剪节省=X.X%
```

### 触发源优先级映射

| triggers 列表内容 | source 值 | significant |
|-------------------|-----------|-------------|
| 含 'motion' | '运动' → source='smart' | True |
| 含 'scene_switch'（无 motion） | '变化' → source='smart' | True |
| 含 'anomaly'（无上述） | '异常' → source='smart' | True |
| 仅 'periodic' | '时间' → source='time' | False |

---

## Step 5: 更新 __init__.py

在 `src/python_test/smart_sampler/__init__.py` 中添加:

```python
from .ml_smart_sampler import MLSmartSampler

__all__ = [
    'MotionDetector',
    'ChangeAnalyzer',
    'ForegroundExtractor',
    'SmartSampler',
    'MLSmartSampler',  # 新增
]
```

---

## 验证方案

1. **单元测试**: 编写 `tests/test_smart_filter.py`
   - 构造合成帧序列（黑屏、白屏、静止、运动、场景切换）
   - 验证 Layer 0 各过滤项逐项生效
   - 验证 Layer 1 各触发器独立触发
   - 验证 Layer 2 峰值检测和 NMS
   - 验证 `sample()` 返回字典包含所有必需键
   - 验证 `frame_index` 全局连续递增

2. **接口兼容性测试**: 
   - 用 SmartFilter 替换 SmartSampler 跑 LivePipeline，验证无报错
   - 对比返回字典 keys 完全一致

3. **性能测试**:
   - 验证 Layer 0 每帧 <0.1ms
   - 验证 Layer 1 每帧 <3ms
   - 验证 Layer 2 每帧 <5ms

4. **依赖检查**: 确认 `imagehash` 已安装（`pip install imagehash`）
