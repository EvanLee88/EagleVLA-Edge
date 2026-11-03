#!/usr/bin/env python3
"""Compare SmolVLA GGUF tokenizer with the Hugging Face tokenizer."""
import argparse
import ctypes
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('library',type=Path);p.add_argument('model',type=Path);p.add_argument('backbone',type=Path)
    args=p.parse_args();lib=ctypes.CDLL(str(args.library.resolve()))
    lib.smolvla_tokenizer_open.argtypes=[ctypes.c_char_p];lib.smolvla_tokenizer_open.restype=ctypes.c_void_p
    lib.smolvla_tokenizer_encode.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint8)]
    lib.smolvla_tokenizer_free.argtypes=[ctypes.c_void_p]
    handle=lib.smolvla_tokenizer_open(str(args.model.resolve()).encode())
    if not handle: raise RuntimeError('cannot load tokenizer')
    tokenizer=AutoTokenizer.from_pretrained(args.backbone,local_files_only=True)
    try:
        for text in ['', 'Pick up the red block and place it in the tray.', 'place it\n', '你好，世界!', 'pick ' * 100, 'a <image> object']:
            value=text if text.endswith('\n') else text+'\n'; expected=tokenizer(value,padding='max_length',truncation=True,max_length=48,return_tensors='np')
            data=text.encode(); ids=(ctypes.c_int32*48)(); mask=(ctypes.c_uint8*48)()
            if lib.smolvla_tokenizer_encode(handle,data,len(data),ids,mask): raise RuntimeError('tokenization failed')
            if not np.array_equal(np.asarray(ids),expected['input_ids'].reshape(-1)) or not np.array_equal(np.asarray(mask),expected['attention_mask'].reshape(-1)):
                raise AssertionError(f'tokenization mismatch: {text!r}\nactual {list(ids)}\nexpected {expected["input_ids"].reshape(-1).tolist()}')
            print(f'passed: {text[:30]!r}')
    finally: lib.smolvla_tokenizer_free(handle)


if __name__=='__main__': main()
