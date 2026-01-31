import os
import sys
import time
import multiprocessing as mp
import numpy as np

# 添加当前目录到路径
sys.path.append(os.getcwd())

from qwen3_tts_gguf.protocol import DecodeRequest, DecodeResult
from qwen3_tts_gguf.workers import decoder_worker_proc

def test_worker_eof():
    print("🧪 [Test] 启动 DecoderWorker EOF 信号测试...")
    
    codes_q = mp.Queue()
    res_q = mp.Queue()
    onnx_path = "model-base/qwen3_tts_decoder.onnx"
    
    # 启动 Worker 进程
    p = mp.Process(target=decoder_worker_proc, args=(codes_q, res_q, onnx_path))
    p.start()
    
    try:
        # 1. 期待收到 READY 信号
        ready_msg = res_q.get(timeout=10)
        print(f"✅ 收到就绪信号: {ready_msg.msg_type}")
        
        # 2. 测试场景：DECODE (隐式结束)
        print("\n--- 场景 1: DECODE (一次性任务) ---")
        dummy_codes = np.zeros((10, 16), dtype=np.int64)
        codes_q.put(DecodeRequest(task_id="task_offline", msg_type="DECODE", codes=dummy_codes))
        
        # 应该先收到音频数据片段
        received_eof = False
        while True:
            res: DecodeResult = res_q.get(timeout=5)
            if res.audio is None:
                print(f"✅ [task_offline] 成功收到 EOF 信号 (audio=None)")
                received_eof = True
                break
            else:
                print(f"   收到音频片段: {len(res.audio)} samples")
        
        if not received_eof: raise Exception("未收到 EOF 信号")

        # 3. 测试场景：DECODE_CHUNK (流式任务)
        print("\n--- 场景 2: DECODE_CHUNK (流式任务) ---")
        # 发送 Chunk 1 (不结束)
        codes_q.put(DecodeRequest(task_id="task_stream", msg_type="DECODE_CHUNK", codes=dummy_codes, is_final=False))
        # 应该只收到音频，没有 EOF
        res = res_q.get(timeout=5)
        print(f"   [Chunk 1] 收到音频: {len(res.audio)} samples")
        
        # 发送 Chunk 2 (结束)
        print("   发送最后一包 (is_final=True)...")
        codes_q.put(DecodeRequest(task_id="task_stream", msg_type="DECODE_CHUNK", codes=dummy_codes, is_final=True))
        
        received_eof = False
        while True:
            res: DecodeResult = res_q.get(timeout=5)
            if res.audio is None:
                print(f"✅ [task_stream] 成功收到 EOF 信号 (audio=None)")
                received_eof = True
                break
            else:
                 print(f"   收到音频片段: {len(res.audio)} samples")
        
        if not received_eof: raise Exception("流式任务结束未收到 EOF 信号")

        print("\n✨ [Success] DecoderWorker EOF 信号逻辑验证通过！")

    finally:
        codes_q.put(None) # 毒丸
        p.join(timeout=2)
        if p.is_alive(): p.terminate()

if __name__ == "__main__":
    test_worker_eof()
