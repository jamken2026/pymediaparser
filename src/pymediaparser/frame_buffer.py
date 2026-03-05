"""帧缓冲池 - 管理待处理帧的缓冲池"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BufferedFrame:
    """缓冲帧数据结构"""
    frame_data: Dict[str, Any]
    timestamp: float


class FrameBuffer:
    """帧缓冲池 - 管理智能采样后的帧，为批量处理做准备
    
    触发条件：
    1. 缓冲区满（帧数 >= max_size）
    2. 帧时间戳跨度超限（跨度 >= max_wait_time）
    
    时间戳跨度：使用帧自身的相对时间戳计算，适用于实时流和回放场景。
    
    时序逻辑：
    - 新帧到达时，先预判入队后的时间跨度
    - 若跨度超限，先返回当前批次，再将新帧入队
    - 确保同一批次内的帧时间连续，便于大模型理解
    """

    def __init__(self, max_size: int = 5, max_wait_time: float = 5.0) -> None:
        self.max_size = max_size
        self.max_wait_time = max_wait_time
        self.buffer: deque[BufferedFrame] = deque(maxlen=max_size)
        logger.debug("FrameBuffer 初始化完成 - max_size=%d, max_wait_time=%.1fs", 
                    max_size, max_wait_time)

    def add_frame(self, frame_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """添加帧到缓冲区，返回就绪的批次（如果有）
        
        时序逻辑：
        1. 预判新帧入队后的时间跨度
        2. 若跨度超限，先返回当前批次
        3. 新帧入队
        4. 检查缓冲区是否满
        
        Args:
            frame_data: 帧数据字典，需包含 'timestamp' 字段
            
        Returns:
            就绪的帧批次，或 None（继续等待）
        """
        if not frame_data:
            return None
        
        new_timestamp = frame_data.get('timestamp', 0.0)
        batch_to_return = None
        
        # 预判：如果新帧入队后时间跨度会超限，先返回当前批次
        if self.buffer:
            first_timestamp = self.buffer[0].timestamp
            predicted_span = new_timestamp - first_timestamp
            
            if predicted_span >= self.max_wait_time:
                # 新帧会导致时间跨度超限 → 先返回当前批次
                batch_to_return = self._prepare_batch()
                logger.info(
                    "[FrameBuffer] 时间跨度超限，先输出批次 - 跨度: %.2fs, 批次帧数: %d",
                    predicted_span, len(batch_to_return) if batch_to_return else 0
                )
        
        # 新帧入队
        buffered_frame = BufferedFrame(
            frame_data=frame_data,
            timestamp=new_timestamp,
        )
        self.buffer.append(buffered_frame)
        logger.debug("帧已添加到缓冲区 - 大小: %d/%d", len(self.buffer), self.max_size)
        
        # 如果上面已返回批次，这里不再检查（新帧刚入队，大概率不满）
        if batch_to_return is not None:
            return batch_to_return
        
        # 检查缓冲区是否满
        if len(self.buffer) >= self.max_size:
            batch = self._prepare_batch()
            logger.info("[FrameBuffer] 缓冲区满，输出批次 - 帧数: %d", len(batch))
            return batch
        
        return None

    def _prepare_batch(self) -> List[Dict[str, Any]]:
        """准备批次，返回所有缓冲帧"""
        result = [f.frame_data for f in self.buffer]
        self.buffer.clear()
        return result

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        logger.info("帧缓冲区已清空")

    def get_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        if not self.buffer:
            return {'size': 0, 'ready': False, 'timestamp_span': 0.0}
        
        timestamp_span = self.buffer[-1].timestamp - self.buffer[0].timestamp
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'timestamp_span': timestamp_span,
            'max_wait_time': self.max_wait_time,
            'ready': len(self.buffer) >= self.max_size or timestamp_span >= self.max_wait_time,
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
