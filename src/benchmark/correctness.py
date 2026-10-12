#!/usr/bin/env python3
import os
import subprocess
import json
import tempfile
import numpy as np
import sys

def numpy_reference(model_dict, input_vals):
    x = np.array(input_vals, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    for layer in model_dict["layers"]:
        l_type = layer["type"]
        if l_type == "dense":
            w = np.array(layer["weights"], dtype=np.float32)
            b = np.array(layer["bias"], dtype=np.float32)
            x = np.matmul(x, w) + b
        elif l_type == "relu":
            x = np.maximum(0.0, x)
        elif l_type == "softmax":
            if x.ndim == 1:
                max_val = np.max(x)
                exps = np.exp(x - max_val)
                x = exps / np.sum(exps)
            else:
                max_val = np.max(x, axis=1, keepdims=True)
                exps = np.exp(x - max_val)
                x = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown layer: {l_type}")
    return x

def run_nier(model_path, input_path):
    try:
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--", "infer", model_path, input_path],
            capture_output=True,
            text=True,
            timeout=5.0
        )
    except subprocess.TimeoutExpired:
        return {"error": "NIER process timed out"}

    if result.returncode != 0:
        return {"error": result.stderr.strip()}

    output_shape = None
    output_data = None

    for line in result.stdout.splitlines():
        if line.startswith("Shape: "):
            try:
                output_shape = json.loads(line[len("Shape: "):])
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse output shape: {e}"}

        elif line.startswith("Data: "):
            try:
                output_data = json.loads(line[len("Data: "):])
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse output data: {e}"}

    if output_shape is None:
        return {"error": f"No output shape found:\n{result.stdout}"}

    if output_data is None:
        return {"error": f"No output data found:\n{result.stdout}"}

    try:
        output = np.asarray(output_data, dtype=np.float32).reshape(output_shape)
        return {"output": output}
    except (ValueError, TypeError) as e:
        return {"error": f"Output shape/data mismatch: {e}"}


def verify_output_properties(final_layer_type, output):
    """Performs mathematical property checks based strictly on the final network layer."""
    if not np.isfinite(output).all():
        return "Output contains non-finite values (NaN or Inf)"

    if final_layer_type == "relu":
        if np.any(output < 0.0):
            return "ReLU output contains negative values (clamping failed)"
    elif final_layer_type == "softmax":
        if np.any(output < 0.0) or np.any(output > 1.0):
            return "Softmax probabilities out of [0, 1] range"
        if output.ndim == 2:
            row_sums = np.sum(output, axis=-1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                return f"Softmax row sums do not equal 1 (got {row_sums})"
        elif output.ndim == 1:
            total_sum = np.sum(output)
            if not np.allclose(total_sum, 1.0, atol=1e-4):
                return f"Softmax sum does not equal 1 (got {total_sum})"
    return None

def main():
    tolerance = 1e-5

    # Comprehensive test suite ensuring every layer is tested in isolation as a terminal output
    test_suites = [
        {
            "name": "1. Single Dense Layer (Linear)",
            "model": {
                "format_version": 1,
                "layers": [{"type": "dense", "weights": [[0.5, -0.5], [1.0, 2.0]], "bias": [0.1, -0.2]}]
            },
            "input": [1.0, 2.0],
            "should_succeed": True
        },
        {
            "name": "2. Isolated ReLU Terminal (Negative Clamping)",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[-1.0, 2.0], [0.5, -1.0]], "bias": [0.0, 0.0]},
                    {"type": "relu"}  # ReLU is terminal here; unmasked!
                ]
            },
            "input": [2.0, -3.0],
            "should_succeed": True
        },
        {
            "name": "3. Multi-layer Feedforward (Dense -> ReLU -> Dense)",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[0.2, 0.8], [0.5, -0.3]], "bias": [0.1, 0.2]},
                    {"type": "relu"},
                    {"type": "dense", "weights": [[0.4], [0.7]], "bias": [0.05]}
                ]
            },
            "input": [1.0, 2.0],
            "should_succeed": True
        },
        {
            "name": "4. Isolated Softmax Terminal (Probability Sums to 1)",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[1.0, 0.0], [0.0, 1.0]], "bias": [1.0, 2.0]},
                    {"type": "softmax"}  # Softmax is terminal here; unmasked!
                ]
            },
            "input": [1.0, 2.0],
            "should_succeed": True
        },
        {
            "name": "5. Large-magnitude Numerical Stability (Softmax)",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[1.0, 0.0], [0.0, 1.0]], "bias": [1000.0, 1001.0]},
                    {"type": "softmax"}
                ]
            },
            "input": [1000.0, 1000.0],
            "should_succeed": True
        },
        {
            "name": "6. Invalid Layer Dimensions (Incompatible Shapes)",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.1, 0.2]},
                    {"type": "dense", "weights": [[1.0, 2.0, 3.0]], "bias": [0.1]}
                ]
            },
            "input": [1.0, 2.0],
            "should_succeed": False
        },
        {
            "name": "7. Input Feature Dimension Mismatch",
            "model": {
                "format_version": 1,
                "layers": [
                    {"type": "dense", "weights": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.1, 0.2]}
                ]
            },
            "input": [1.0, 2.0, 3.0],
            "should_succeed": False
        }
    ]

    all_passed = True
    print(f"{'Test Case':<55} | {'E_max (Max Abs Error)':<22} | {'Status'}")
    print("-" * 90)

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, test in enumerate(test_suites):
            m_path = os.path.join(tmpdir, f"model_{idx}.json")
            i_path = os.path.join(tmpdir, f"input_{idx}.json")

            with open(m_path, "w") as f:
                json.dump(test["model"], f)
            with open(i_path, "w") as f:
                json.dump({"input": test["input"]}, f)

            res = run_nier(m_path, i_path)

            if not test["should_succeed"]:
                if "error" in res:
                    print(f"{test['name']:<55} | {'N/A (Correctly Rejected)':<22} | PASSED ✅")
                else:
                    all_passed = False
                    print(f"{test['name']:<55} | {'N/A (Expected Failure)':<22} | FAILED ❌")
            else:
                if "error" in res:
                    all_passed = False
                    print(f"{test['name']:<55} | {'ERROR':<22} | FAILED ❌ ({res['error']})")
                else:
                    try:
                        # 1. Generate unflattened reference output
                        ref_out_full = numpy_reference(test["model"], test["input"])
                        nier_output = res["output"]

                        if nier_output.size == 0 or ref_out_full.size == 0:
                            raise ValueError("Inference returned an empty output array")

                        # 2. Strict Unflattened Shape Contract Check (FIX 1)
                        if nier_output.shape != ref_out_full.shape:
                            raise ValueError(f"Shape contract violation: reference shape {ref_out_full.shape} vs NIER shape {nier_output.shape}")

                        # Determine final layer type
                        final_layer_type = test["model"]["layers"][-1]["type"]

                        # 3. Independent Property Assertions on unflattened output
                        prop_err = verify_output_properties(final_layer_type, nier_output)
                        if prop_err:
                            raise ValueError(f"Property violation: {prop_err}")

                        # 4. Numerical Comparison against NumPy Reference (flattened for element-wise max error)
                        ref_out_flat = ref_out_full.flatten()
                        nier_out_flat = nier_output.flatten()

                        e_max = np.max(np.abs(nier_out_flat - ref_out_flat))
                        passed = e_max < tolerance

                        if not passed:
                            all_passed = False
                            status = "FAILED ❌"
                        else:
                            status = "PASSED ✅"

                        print(f"{test['name']:<55} | {e_max:<22.2e} | {status}")
                    except Exception as e:
                        all_passed = False
                        print(f"{test['name']:<55} | {'EXCEPTION':<22} | FAILED ❌ ({e})")

    print("-" * 90)
    if all_passed:
        print("\nAll correctness, unflattened shape contracts, terminal activation checks, and error tests passed! 🎉")
        sys.exit(0)
    else:
        print("\nSome correctness validation tests failed! ⚠️")
        sys.exit(1)

if __name__ == "__main__":
    main()