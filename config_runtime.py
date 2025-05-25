"""
config_runtime.py
=================

集中放所有 **运行期加速选项**。
*默认全部打开*，但是你在训练脚本里可以一键关闭：

>>> from config_runtime import RF
>>> RF.disable_all()          # 关闭所有加速
>>> RF.batch_processor = False # 单独再开某一项
"""

from dataclasses import dataclass, field

@dataclass
class _Flags:
    # ---------- ① 批量前向 ----------
    batch_sensor: bool = True       # sensor forward 已经改过
    batch_processor: bool = True
    batch_emitter: bool = False

    # ---------- ② torch.compile ----------
    use_compile: bool = True        # 编译 Shared-Net，mode 见 compile_mode
    compile_mode: str = "reduce-overhead"   # 或 "max-autotune"

    # ---------- ③ CUDA Graph ----------
    use_cuda_graph: bool = True     # 仅 GPU 推理阶段生效

    # ---------- ④ 内存 / 精度 ----------
    use_channels_last: bool = True  # Linear 网络转 NHWC，访存更顺
    use_fp16: bool = True           # autocast + 权重 half
    # ---------- ⑥ Shared-Transformer ----------
    use_shared_tx: bool   = False   # 把 N 个 CogUnit 当序列一次性跑
    shared_tx_layers: int = 4      # 堆多少层
    shared_tx_heads: int  = 8      # Multi-Head 个数
    shared_tx_interval: int = 10    # 每隔多少步跑一次（>1 可省 Python）
    # ---------- ⑤ 全局开关 ----------
    def disable_all(self):
        """一次性关掉所有加速路径（Debug / CPU 训练用）。"""
        for k in vars(self).keys():
            if not k.startswith("_") and isinstance(getattr(self, k), bool):
                setattr(self, k, False)

# module-level 单例，外部用 RF.XXX 访问
RF: _Flags = _Flags()


"""
现在还是cpu环境，以后换环境，记得加装
# Flash-Attn 2 + Transformer-Engine（FP16 / BF16 / H100 上 FP8 都支持）
pip install flash-attn --no-build-isolation
pip install transformer-engine
"""