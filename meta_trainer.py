import torch
import torch.nn.functional as F
from torch import optim
from higher import innerloop_ctx

class MetaTrainer:
    """
    基于 higher 的 MAML 风格元学习 trainer。
    tasks_batch: List of dicts, 每个 dict 包含
        'support': List of (state_vec, action, label)
        'query':   List of (state_vec, action, label)
    """

    def __init__(self, processor, lr_inner=1e-3, lr_meta=1e-4, device="cpu"):
        # processor 应当是一个 nn.Module，比如 ImmuneProcessor
        self.processor = processor.to(device)
        self.device    = device
        self.lr_inner  = lr_inner
        # meta‐optimizer 只更新 processor.parameters()
        self.meta_opt  = optim.Adam(self.processor.parameters(), lr=lr_meta)

    def meta_update(self, tasks_batch):
        if not tasks_batch:
            return

        # 用 tensor 初始化，这样加 loss_q 后会保留梯度
        meta_loss = torch.tensor(0.0, device=self.device)

        for task in tasks_batch:
            support = task['support']
            query   = task['query']

            # inner‐loop 优化器实例化
            inner_opt = optim.SGD(self.processor.parameters(), lr=self.lr_inner)

            # higher 上下文
            with innerloop_ctx(self.processor, inner_opt, device=self.device) as (fmodel, diffopt):
                # —— inner‐loop on support ——
                s_states = torch.stack([t[0] for t in support]).to(self.device)        # (K, D)
                s_labels = torch.tensor([t[2] for t in support], dtype=torch.long, device=self.device)  # (K,)
                # —— inner‐loop on support ——
                logits_s = fmodel.classifier(s_states)
                loss_s = F.cross_entropy(logits_s, s_labels)

                diffopt.step(loss_s)


                q_states = torch.stack([t[0] for t in query]).to(self.device)        # (K', D)
                q_labels = torch.tensor([t[2] for t in query], dtype=torch.long, device=self.device)  # (K',)


                logits_q = fmodel.classifier(q_states)
                loss_q = F.cross_entropy(logits_q, q_labels)
                meta_loss = meta_loss + loss_q

        # 平均并做一次 meta 更新
        meta_loss = meta_loss / len(tasks_batch)
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
