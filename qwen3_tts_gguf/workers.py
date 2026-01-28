import os
import time
import numpy as np
import queue
import soundfile as sf

def wav_writer_proc(record_queue, filename, sample_rate=24000):
    abs_filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    try:
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    except:
        abs_filename = abs_filename.replace(".wav", f"_{int(time.time())}.wav")
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    try:
        while True:
            chunk = record_queue.get()
            if chunk is None: break
            f.write(chunk.flatten().astype(np.float32))
            f.flush()
    except: pass
    finally: f.close()

def decoder_worker_proc(codes_queue, pcm_queue, mouth_onnx_path, record_queue=None):
    """
    解码工人：极致精简位对齐方案。
    由于 LEFT_CONTEXT=72 已经让模型达到了相位稳定性，不再需要昂贵的 NCC 匹配。
    """
    import onnxruntime as ort
    
    LEFT_CONTEXT = 72    
    RIGHT_CONTEXT = 4    
    UPSAMPLE_RATE = 1920 

    sess = ort.InferenceSession(mouth_onnx_path, providers=['CPUExecutionProvider'])
    history_codes = np.zeros((0, 16), dtype=np.int64)
    
    try:
        while True:
            task = codes_queue.get()
            if task is None:
                pcm_queue.put(None)
                if record_queue: record_queue.put(None)
                break
            
            working_codes = np.array(task).astype(np.int64)
            
            # 1. 全量解码：此时输出在数学上是分片连贯的
            c_in = working_codes[np.newaxis, ...].astype(np.int64)
            full_audio = sess.run(None, {'audio_codes': c_in})[0].squeeze().astype(np.float32)
            
            # 2. 精准定位“干货区”
            actual_history_steps = min(len(history_codes), LEFT_CONTEXT)
            start_idx = actual_history_steps * UPSAMPLE_RATE
            
            current_chunk_steps = len(working_codes) - actual_history_steps - RIGHT_CONTEXT
            n_samples = current_chunk_steps * UPSAMPLE_RATE
            
            # 3. 直接切割：因为有足够的上下文，切割点在物理上已经是连续的
            output_chunk = full_audio[start_idx : start_idx + n_samples]

            # 4. 交付
            pcm_queue.put(output_chunk.copy())
            if record_queue: record_queue.put(output_chunk.copy())
            
            # 5. 更新状态用于下一次拼接
            history_codes = working_codes[:actual_history_steps + current_chunk_steps][-LEFT_CONTEXT:]

    except Exception as e:
        print(f"  [DecoderWorker] 异常: {e}")
    finally: pass

def speaker_worker_proc(pcm_queue, sample_rate=24000):
    import sounddevice as sd
    state = {"current_data": np.zeros((0, 1), dtype=np.float32), "started": False, "prefill": 4800}
    def audio_callback(outdata, frames, time_info, status):
        while True:
            try:
                new_item = pcm_queue.get_nowait()
                if new_item is None: break
                state["current_data"] = np.concatenate([state["current_data"], new_item.reshape(-1, 1).astype(np.float32)], axis=0)
            except queue.Empty: break
        if not state["started"]:
            if len(state["current_data"]) >= state["prefill"]: state["started"] = True
            else: outdata.fill(0); return
        avail = len(state["current_data"])
        to_copy = min(avail, frames)
        if to_copy > 0:
            outdata[:to_copy] = state["current_data"][:to_copy]
            state["current_data"] = state["current_data"][to_copy:]
        if to_copy < frames:
            outdata[to_copy:].fill(0); state["started"] = False
    try:
        with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=1024):
            while True: time.sleep(1)
    except: pass
