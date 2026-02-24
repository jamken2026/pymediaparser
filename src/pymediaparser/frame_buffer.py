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
    is_significant: bool
    added_time: float
    
    def __post_init__(self):
        self.added_time = time.time()

class FrameBuffer:
    """帧缓冲池 - 管理智能采样后的帧，为批量处理做准备"""

    def __init__(self, max_size: int = 5, min_significant_frames: int = 2, max_wait_time: float = 2.0) -> None:
        self.max_size = max_size
        self.min_significant = min_significant_frames
        self.max_wait_time = max_wait_time
        self.buffer: deque[BufferedFrame] = deque(maxlen=max_size)
        self.last_batch_time: float = time.time()
        logger.debug("FrameBuffer 初始化完成")

    def add_frame(self, frame_data: Dict[str, Any]) -> None:
        if not frame_data:
            return
            
        buffered_frame = BufferedFrame(
            frame_data=frame_data,
            timestamp=frame_data.get('timestamp', time.time()),
            is_significant=frame_data.get('significant', False),
            added_time=time.time()
        )
        
        self.buffer.append(buffered_frame)
        logger.debug("帧已添加到缓冲区 - 显著: %s, 缓冲区大小: %d/%d", 
                    buffered_frame.is_significant, len(self.buffer), self.max_size)

    def get_ready_batch(self) -> Optional[List[Dict[str, Any]]]:
        if not self.buffer:
            return None
            
        current_time = time.time()
        time_elapsed = current_time - self.last_batch_time
        
        significant_frames = [f for f in self.buffer if f.is_significant]
        significant_count = len(significant_frames)
        
        should_process = (
            significant_count >= self.min_significant or
            len(self.buffer) >= self.max_size or
            time_elapsed >= self.max_wait_time
        )
        
        if should_process:
            batch = self._prepare_batch()
            self.last_batch_time = current_time
            logger.info("准备批处理 - 显著帧: %d/%d, 总帧数: %d, 等待时间: %.1fs", 
                       significant_count, self.min_significant, len(batch), time_elapsed)
            return batch
        
        return None

    def _prepare_batch(self) -> List[Dict[str, Any]]:
        significant_frames = [f for f in self.buffer if f.is_significant]
        normal_frames = [f for f in self.buffer if not f.is_significant]
        
        batch_frames = significant_frames[:3]
        remaining_slots = 3 - len(batch_frames)
        
        if remaining_slots > 0 and normal_frames:
            recent_normal = sorted(normal_frames, key=lambda x: x.timestamp, reverse=True)
            batch_frames.extend(recent_normal[:remaining_slots])
        
        batch_frames.sort(key=lambda x: x.timestamp)
        result = [f.frame_data for f in batch_frames]
        
        processed_timestamps = {f.timestamp for f in batch_frames}
        self.buffer = deque([f for f in self.buffer if f.timestamp not in processed_timestamps], 
                           maxlen=self.max_size)
        
        return result

    def clear(self) -> None:
        self.buffer.clear()
        self.last_batch_time = time.time()
        logger.info("帧缓冲区已清空")

    def get_status(self) -> Dict[str, Any]:
        if not self.buffer:
            return {'size': 0, 'significant_count': 0, 'ready': False}
            
        significant_count = sum(1 for f in self.buffer if f.is_significant)
        time_waiting = time.time() - self.last_batch_time
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'significant_count': significant_count,
            'min_required': self.min_significant,
            'time_waiting': time_waiting,
            'max_wait_time': self.max_wait_time,
            'ready': (significant_count >= self.min_significant or 
                     len(self.buffer) >= self.max_size or
                     time_waiting >= self.max_wait_time)
        }

    def flush(self) -> Optional[List[Dict[str, Any]]]:
        """强制清空缓冲区，返回所有缓存的帧。
        
        用于停止时处理剩余帧。
        
        Returns:
            如果有缓存帧，返回帧数据列表；否则返回 None
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
        return len(self.buffer) > 0
