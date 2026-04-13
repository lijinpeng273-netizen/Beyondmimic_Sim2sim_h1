"""ONNX 策略模型元数据查看工具。

用法：
    python tools/read_onnx.py <onnx_path>

功能：
    从 ONNX 模型的 metadata_props 中提取并打印所有嵌入的训练配置信息，
    包括关节名称、默认位置、PD 增益、动作缩放等参数。

示例：
    python tools/read_onnx.py models/dance_walk_01.onnx
"""

import sys
import onnx

def main():
    if len(sys.argv) < 2:
        print("用法: python read_onnx.py <onnx_file_path>")
        print("示例: python read_onnx.py models/dance_walk_01.onnx")
        sys.exit(1)

    model = onnx.load(sys.argv[1])
    
    print(f"=== ONNX 模型元数据: {sys.argv[1]} ===\n")
    
    # 打印输入/输出信息
    print("--- 模型输入 ---")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: shape={shape}")
    
    print("\n--- 模型输出 ---")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: shape={shape}")
    
    print("\n--- 训练元数据 ---")
    for prop in model.metadata_props:
        # 截断过长的值以便阅读
        val = prop.value
        if len(val) > 200:
            val = val[:200] + "..."
        print(f"  {prop.key}: {val}")

if __name__ == "__main__":
    main()
