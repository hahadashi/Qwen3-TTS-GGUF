"""
AsyncAudioDecoder - 异步并行音频解码器

核心功能:
- 使用独立线程执行音频解码
- codes 累积后非阻塞提交解码任务
- 主线程可以继续生成下一帧
- 通过 Queue 获取已完成的音频

架构:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Main Thread │ ──► │ CodesQueue  │ ──► │ DecodeThread│
│ (生成 codes) │     │             │     │ (解码音频)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ AudioQueue  │
                                       │ (音频结果)   │
                                       └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ Main Thread │
                                       │ (收集音频)   │
                                       └─────────────┘

作者: Claude
日期: 2026-03-25
"""

import threading
import queue
from typing import Optional, Callable, List
import torch


class AsyncAudioDecoder:
    """
    异步并行音频解码器

    使用独立线程执行音频解码，主线程可以继续生成下一帧。
    通过 Queue 传递 codes 和音频数据。

    使用方式:
        >>> def decode_fn(chunk_codes):
        ...     return decoder.decode(chunk_codes)
        >>>
        >>> async_decoder = AsyncAudioDecoder(decode_fn=decode_fn, chunk_size=12)
        >>>
        >>> # 主线程生成循环
        >>> for codes in generate_loop():
        ...     async_decoder.submit_frame(codes)
        ...     if buffer_full:
        ...         async_decoder.submit_chunk()  # 非阻塞
        >>>
        >>> # 获取最终音频
        >>> audio = async_decoder.flush()
        >>> async_decoder.shutdown()
    """

    def __init__(
        self,
        decode_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        chunk_size: int = 12,
        warmup_fn: Optional[Callable[[], None]] = None,
    ):
        """
        初始化异步解码器

        Args:
            decode_fn: 解码函数，接收 List[codes] 返回 audio tensor
            chunk_size: 每个 chunk 的帧数
            warmup_fn: 可选的预热函数（用于初始化解码器历史）
        """
        self.decode_fn = decode_fn
        self.chunk_size = chunk_size
        self.warmup_fn = warmup_fn

        # 线程安全队列
        self._decode_queue: queue.Queue = queue.Queue()  # (chunk_codes, chunk_id)
        self._audio_queue: queue.Queue = queue.Queue()   # (audio, chunk_id, error)

        # 状态 (需要锁保护)
        self._codes_buffer: List[torch.Tensor] = []
        self._audio_chunks: dict = {}  # chunk_id -> audio
        self._chunk_counter: int = 0
        self._next_output_chunk: int = 0
        self._error: Optional[Exception] = None
        self._lock = threading.Lock()

        # 工作线程
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_worker()

    def _start_worker(self) -> None:
        """启动工作线程"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # 执行预热
        if self.warmup_fn is not None:
            # 预热在工作线程中执行，确保线程安全
            self._decode_queue.put(("warmup", None, -1))

    def _worker_loop(self) -> None:
        """工作线程主循环"""
        while self._running:
            try:
                item = self._decode_queue.get(timeout=0.1)
                if item is None:
                    continue

                msg_type, chunk_codes, chunk_id = item

                if msg_type == "warmup":
                    # 预热
                    try:
                        if self.warmup_fn is not None:
                            self.warmup_fn()
                        self._audio_queue.put((None, -1, None))
                    except Exception as e:
                        self._audio_queue.put((None, -1, e))

                elif msg_type == "decode":
                    # 解码
                    try:
                        audio = self.decode_fn(chunk_codes)
                        self._audio_queue.put((audio, chunk_id, None))
                    except Exception as e:
                        self._audio_queue.put((None, chunk_id, e))

                elif msg_type == "shutdown":
                    break

            except queue.Empty:
                continue
            except Exception as e:
                # 记录错误
                with self._lock:
                    self._error = e

    def submit_frame(self, codes: torch.Tensor) -> None:
        """
        提交一帧 codes（非阻塞）

        Args:
            codes: [1, 16] 单帧 codes
        """
        with self._lock:
            self._codes_buffer.append(codes)

    def submit_chunk(self) -> int:
        """
        触发当前 chunk 解码（非阻塞）

        Returns:
            chunk_id: chunk 编号，用于后续获取结果
            -1 表示没有数据需要提交
        """
        with self._lock:
            if not self._codes_buffer:
                return -1

            chunk = list(self._codes_buffer)
            self._codes_buffer = []
            chunk_id = self._chunk_counter
            self._chunk_counter += 1

            self._decode_queue.put(("decode", chunk, chunk_id))
            return chunk_id

    def get_audio(self, timeout: float = 0.0) -> Optional[torch.Tensor]:
        """
        尝试获取已完成的音频

        Args:
            timeout: 等待超时（秒），0 表示非阻塞

        Returns:
            audio: 解码完成的音频，None 表示没有可用音频

        Raises:
            RuntimeError: 如果解码线程发生错误
        """
        # 检查错误
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"解码线程错误: {self._error}")

        try:
            audio, chunk_id, error = self._audio_queue.get(timeout=timeout)
            if error is not None:
                raise RuntimeError(f"解码错误 (chunk {chunk_id}): {error}")
            if audio is not None:
                with self._lock:
                    self._audio_chunks[chunk_id] = audio
                return audio
            return None
        except queue.Empty:
            return None

    def get_all_audio(self) -> torch.Tensor:
        """
        获取所有已完成的音频，按顺序拼接

        Returns:
            audio: [samples] 拼接后的音频

        Raises:
            RuntimeError: 如果解码线程发生错误
        """
        with self._lock:
            # 检查错误
            if self._error is not None:
                raise RuntimeError(f"解码线程错误: {self._error}")

            # 收集所有可用音频
            while True:
                try:
                    audio, chunk_id, error = self._audio_queue.get_nowait()
                    if error is not None:
                        raise RuntimeError(f"解码错误 (chunk {chunk_id}): {error}")
                    if audio is not None:
                        self._audio_chunks[chunk_id] = audio
                except queue.Empty:
                    break

            # 按顺序输出
            audios = []
            for i in range(self._next_output_chunk, self._chunk_counter):
                if i in self._audio_chunks:
                    audios.append(self._audio_chunks[i])
                    self._next_output_chunk = i + 1

            return torch.cat(audios) if audios else torch.tensor([], device=audios[0].device if audios else "cpu")

    def flush(self) -> torch.Tensor:
        """
        刷新剩余 codes 并等待所有解码完成

        Returns:
            audio: [samples] 所有音频拼接后的结果
        """
        # 提交剩余 codes
        self.submit_chunk()

        # 等待所有解码完成
        self._wait_all()

        # 返回所有音频
        return self.get_all_audio()

    def _wait_all(self, timeout_per_chunk: float = 10.0) -> None:
        """
        等待所有解码任务完成

        Args:
            timeout_per_chunk: 每个 chunk 的等待超时
        """
        while True:
            with self._lock:
                expected = self._chunk_counter
                received = len(self._audio_chunks)
                error = self._error

            # 检查错误
            if error is not None:
                raise RuntimeError(f"解码线程错误: {error}")

            # 检查是否所有 chunk 都已接收
            if received >= expected:
                break

            # 等待下一个 chunk
            try:
                self.get_audio(timeout=timeout_per_chunk)
            except RuntimeError:
                # 错误已在上面检查
                break

    def shutdown(self) -> None:
        """关闭解码器，释放资源"""
        self._running = False
        if self._worker_thread is not None:
            self._decode_queue.put(("shutdown", None, -1))
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    @property
    def pending_chunks(self) -> int:
        """待处理的 chunk 数量"""
        with self._lock:
            return self._chunk_counter - len(self._audio_chunks)

    @property
    def is_running(self) -> bool:
        """解码器是否正在运行"""
        return self._running
