"""
88-Benchmark-ONNX.py
对比测试旧版嘴巴 (Non-streaming) 与新版嘴巴 (Stateful Streaming) 的推理速度。
测试场景：125 帧音频码 (约 10s 音频)。
"""
import os
import time
import numpy as np
import onnxruntime as ort

def main():
    # 配置
    OLD_ONNX_PATH = "model-base/qwen3_tts_decoder.onnx"
    NEW_ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
    
    TOTAL_FRAMES = 125
    Q = 16
    NUM_LAYERS = 8      # 重构版层数
    NUM_HEADS = 16      # 重构版头数
    HEAD_DIM = 64       # 重构版头维度
    
    # 准备随机输入
    # Shape: [1, 125, 16]
    dummy_codes = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    
    # 初始化 session 选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 限定单线程测试以排除多核干扰 (可选，但为了公平建议对比)
    # sess_options.intra_op_num_threads = 1
    
    # 1. 测试旧版嘴巴 (一次性推理)
    if os.path.exists(OLD_ONNX_PATH):
        print(f"📦 正在加载旧版模型: {OLD_ONNX_PATH}")
        old_sess = ort.InferenceSession(OLD_ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
        
        # 预热一次
        old_sess.run(None, {'audio_codes': dummy_codes})
        
        print("🔥 正在测试旧版一次性速度...")
        t_start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            _ = old_sess.run(None, {'audio_codes': dummy_codes})
        t_end = time.perf_counter()
        avg_old_once = (t_end - t_start) / iters
        print(f"   [OLD ONCE] 平均耗时: {avg_old_once*1000:.2f} ms")
    else:
        print(f"⚠️ 未找到旧版模型: {OLD_ONNX_PATH}，跳过对比。")
        avg_old_once = 0

    # 2. 加载新版模型
    print(f"\n📦 正在加载新版模型: {NEW_ONNX_PATH}")
    new_sess = ort.InferenceSession(NEW_ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 初始化状态
    def init_new_states():
        pre_conv_h = np.zeros((1, 512, 0), dtype=np.float32)
        latent_buf = np.zeros((1, 1024, 0), dtype=np.float32)
        conv_h = np.zeros((1, 1024, 0), dtype=np.float32)
        pkv = []
        for _ in range(NUM_LAYERS):
            pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))  # K
        for _ in range(NUM_LAYERS):
            pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))  # V
        return pre_conv_h, latent_buf, conv_h, pkv

    def run_new_streaming(chunk_size):
        pre_conv, latent, conv, pkv = init_new_states()
        output_names = [out.name for out in new_sess.get_outputs()]
        
        t_total = 0
        for i in range(0, TOTAL_FRAMES, chunk_size):
            chunk = dummy_codes[:, i:i+chunk_size, :]
            is_last = np.array([1.0 if i + chunk_size >= TOTAL_FRAMES else 0.0], dtype=np.float32)
            
            feed = {
                "audio_codes": chunk,
                "is_last": is_last,
                "pre_conv_history": pre_conv,
                "latent_buffer": latent,
                "conv_history": conv,
            }
            for j in range(NUM_LAYERS):
                feed[f"past_key_{j}"] = pkv[j]
                feed[f"past_value_{j}"] = pkv[NUM_LAYERS + j]
                
            ts = time.perf_counter()
            outputs = new_sess.run(output_names, feed)
            te = time.perf_counter()
            t_total += (te - ts)
            
            # 更新状态
            pre_conv = outputs[2]
            latent = outputs[3]
            conv = outputs[4]
            for j in range(NUM_LAYERS):
                pkv[j] = outputs[5 + j]
                pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
        return t_total

    # ==========================
    # 3. 开始新版对比实验
    # ==========================
    
    # 3.1 新版一次性推理 (chunk_size = 125)
    print("🔥 正在测试新版一次性速度 (N=125)...")
    run_new_streaming(TOTAL_FRAMES) # 预热
    t_start = time.perf_counter()
    iters = 5
    for _ in range(iters):
        run_new_streaming(TOTAL_FRAMES)
    avg_new_once = (time.perf_counter() - t_start) / iters
    print(f"   [NEW ONCE] 平均耗时: {avg_new_once*1000:.2f} ms")

    # 3.2 新版 12 帧流式 (chunk_size = 12)
    print("🔥 正在测试新版流式速度 (Chunk=12)...")
    run_new_streaming(12) # 预热
    t_start = time.perf_counter()
    iters = 5
    for _ in range(iters):
        run_new_streaming(12)
    avg_new_s12 = (time.perf_counter() - t_start) / iters
    print(f"   [NEW S12]  总均耗时: {avg_new_s12*1000:.2f} ms (单跳均耗: {avg_new_s12*1000/(TOTAL_FRAMES/12):.2f} ms)")

    # 3.3 新版 25 帧流式 (chunk_size = 25)
    print("🔥 正在测试新版流式速度 (Chunk=25)...")
    run_new_streaming(25) # 预热
    t_start = time.perf_counter()
    iters = 5
    for _ in range(iters):
        run_new_streaming(25)
    avg_new_s25 = (time.perf_counter() - t_start) / iters
    print(f"   [NEW S25]  总均耗时: {avg_new_s25*1000:.2f} ms (单跳均耗: {avg_new_s25*1000/(TOTAL_FRAMES/25):.2f} ms)")

    # ==========================
    # 4. 汇总报告
    # ==========================
    print("\n" + "="*50)
    print("📊 性能对比汇总报告 (125 帧 / ~10s 音频)")
    print("="*50)
    if avg_old_once > 0:
        print(f"1. 旧版嘴巴 (一次性):  {avg_old_once*1000:7.2f} ms | RTF: {avg_old_once/10:.4f}")
    print(f"2. 新版嘴巴 (一次性):  {avg_new_once*1000:7.2f} ms | RTF: {avg_new_once/10:.4f}")
    print(f"3. 新版嘴巴 (12帧流式): {avg_new_s12*1000:7.2f} ms | RTF: {avg_new_s12/10:.4f}")
    print(f"4. 新版嘴巴 (25帧流式): {avg_new_s25*1000:7.2f} ms | RTF: {avg_new_s25/10:.4f}")
    print("-" * 50)
    
    if avg_old_once > 0:
        speedup = avg_old_once / avg_new_once
        print(f"🚀 新版(一次性) 相比旧版 加速了: {speedup:.2fx}")
        print(f"💡 流式开销比 (S25 vs Once): {(avg_new_s25/avg_new_once - 1)*100:.1f}%")
        print(f"💡 流式开销比 (S12 vs Once): {(avg_new_s12/avg_new_once - 1)*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()
