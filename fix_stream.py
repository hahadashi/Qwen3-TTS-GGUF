
import os

path = r'd:\qwen3-tts\qwen3_tts_gguf\stream.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Clone
target1 = '''            lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
            return self._post_process(text, pdata, lout, cfg=cfg)'''
replace1 = '''            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)'''

# 2. Update Custom
target2 = '''            lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
            return self._post_process(text, pdata, lout, cfg=cfg)'''
# target2 is same as target1 string-wise but let's do a global replace for these specific patterns if they match

content = content.replace(target1, replace1)

with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Done")
