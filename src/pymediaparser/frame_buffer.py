"""帧缓冲池 - 管理待处理帧的缓冲池"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BufferedFrame:
    frame_data: Dict[str, Any]
    timestamp: float
    added_time: float
    
    def __post_init__(self):
        self.added_time = time.time()


class FrameBuffer:
    """帧缓冲池 - 管理智能采样后的帧，为批量处理做准备
    
    触发条件：
    1. 缓冲区满（帧数 >= max_size）
    2. 超时（等待时间 >= max_wait_time）
    """

    def __init__(self, max_size: int = 5, max_wait_time: float = 2.0) -> None:
        self.max_size = max_size
        self.max_wait_time = max_wait_time
        self.buffer: deque[BufferedFrame] = deque(maxlen=max_size)
        # 第一帧入队时间，用于计算等待超时；None 表示缓冲区为空
        self._first_frame_time: Optional[float] = None
        logger.debug("FrameBuffer 初始化完成 - max_size=%d, max_wait_time=%.1fs", 
                    max_size, max_wait_time)

    def add_frame(self, frame_data: Dict[str, Any]) -> None:
        """添加帧到缓冲区"""
        if not frame_data:
            return
            
        # 第一帧入队时记录时间
        if not self.buffer:
            self._first_frame_time = time.time()
            
        buffered_frame = BufferedFrame(
            frame_data=frame_data,
            timestamp=frame_data.get('timestamp', time.time()),
            added_time=time.time()
        )
        
        self.buffer.append(buffered_frame)
        logger.debug("帧已添加到缓冲区 - 大小: %d/%d", len(self.buffer), self.max_size)

    def get_ready_batch(self) -> Optional[List[Dict[str, Any]]]:
        """检查是否满足批处理条件，返回就绪的帧批次"""
        if not self.buffer:
            logger.debug("[FrameBuffer] 缓冲区为空，无法批处理")
            return None
            
        current_time = time.time()
        first_frame_time = self._first_frame_time or current_time
        time_elapsed = current_time - first_frame_time
        
        # 触发条件：缓冲区满 或 超时
        should_process = (
            len(self.buffer) >= self.max_size or
            time_elapsed >= self.max_wait_time
        )
        
        logger.debug(
            "[FrameBuffer] 批处理检查 - 缓冲区: %d/%d, 等待时间: %.1f/%.1fs",
            len(self.buffer), self.max_size, time_elapsed, self.max_wait_time
        )
        
        if should_process:
            batch = self._prepare_batch()
            # 缓冲区已清空，重置第一帧时间
            self._first_frame_time = None
            
            logger.info("准备批处理 - 总帧数: %d, 等待时间: %.1fs", len(batch), time_elapsed)
            return batch
        
        logger.debug("[FrameBuffer] 批处理条件未满足，继续等待")
        return None

    def _prepare_batch(self) -> List[Dict[str, Any]]:
        """准备批次，返回所有缓冲帧"""
        result = [f.frame_data for f in self.buffer]
        self.buffer.clear()
        return result

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        self._first_frame_time = None
        logger.info("帧缓冲区已清空")

    def get_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        if not self.buffer:
            return {'size': 0, 'ready': False}
            
        first_frame_time = self._first_frame_time or time.time()
        time_waiting = time.time() - first_frame_time
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'time_waiting': time_waiting,
            'max_wait_time': self.max_wait_time,
            'ready': len(self.buffer) >= self.max_size or time_waiting >= self.max_wait_time,
        }

    def flush(self) -> Optional[List[Dict[str, Any]]]:
        """强制清空缓冲区，返回所有缓存的帧
        
        用于停止时处理剩余帧。
        """
        if not self.buffer:
            return None
            
        frames_data = [f.frame_data for f in self.buffer]
        self.clear()
        logger.info("强制清空缓冲区 - 帧数: %d", len(frames_data))
        return frames_data

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        """FrameBuffer 对象始终为 True（表示批处理功能启用）"""
        return True
