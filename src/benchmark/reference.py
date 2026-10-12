#!/usr/bin/env python3
import sys
import json
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def softmax(x):
    if x.ndim == 1:
        max_val = np.max(x)
        exps = np.exp(x - max_val)
        return exps / np.sum(exps)
    elif x.ndim == 2:
        max_val = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - max_val)
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        raise ValueError("Softmax supports 1D or 2D arrays")

def run_inference(model_path, input_path):
    with open(model_path, 'r') as f:
        model_config = json.load(f)

    with open(input_path, 'r') as f:
        input_config = json.load(f)

    x = np.array(input_config["input"], dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    for idx, layer in enumerate(model_config["layers"]):
        l_type = layer["type"]
        if l_type == "dense":
            w = np.array(layer["weights"], dtype=np.float32)
            b = np.array(layer["bias"], dtype=np.float32)
            x = np.matmul(x, w) + b
        elif l_type == "relu":
            x = relu(x)
        elif l_type == "softmax":
            x = softmax(x)
        else:
            raise ValueError(f"Unknown layer type: {l_type}")

    # Print raw JSON array list for direct comparison
    print(json.dumps(x.tolist()))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 reference.py <model_path> <input_path>")
        sys.exit(1)

    run_inference(sys.argv[1], sys.argv[2])