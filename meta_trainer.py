from higher import innerloop_ctx
import torch
import torch.nn.functional as F

class MetaTrainer:
    def __init__(self, processor, lr_inner=1e-3, lr_meta=1e-4, device="cpu"):
        # 把 processor 放到指定 device
        self.processor = processor.to(device)
        self.device    = device
        self.lr_inner  = lr_inner
        # 元优化器
        self.meta_opt  = torch.optim.Adam(self.processor.parameters(), lr=lr_meta)

    def meta_update(self, tasks_batch):
        """
        tasks_batch: List of dicts, 每个 dict 包含
            'support': List of (state, action, label)
            'query':   List of (state, action, label)
        """
        if not tasks_batch:
            return

        # 用 Python float 初始化，第一轮加 loss_q 会变成带梯度的 Tensor
        meta_loss = 0.0

        for task in tasks_batch:
            support = task['support']
            query   = task['query']

            # 每个任务里复用一个 inner‐loop 优化器实例
            inner_opt = torch.optim.SGD(self.processor.parameters(),
                                        lr=self.lr_inner)
            with innerloop_ctx(self.processor,
                               inner_opt,
                               device=self.device) as (fmodel, diffopt):

                # —— inner‐loop on support ——
                # 只取每条 (state, action, label) 的 state 和 label
                s_states = torch.stack([t[0] for t in support]) \
                                   .to(self.device)        # (K, D)
                s_labels = torch.tensor([t[2] for t in support],
                                        dtype=torch.long,
                                        device=self.device)  # (K,)
                loss_s = F.cross_entropy(
                    fmodel._extract_features(s_states),
                    s_labels
                )
                diffopt.step(loss_s)

                # —— evaluate on query ——
                q_states = torch.stack([t[0] for t in query]) \
                                   .to(self.device)        # (K', D)
                q_labels = torch.tensor([t[2] for t in query],
                                        dtype=torch.long,
                                        device=self.device)  # (K',)
                loss_q = F.cross_entropy(
                    fmodel._extract_features(q_states),
                    q_labels
                )

                # 累积 meta loss
                meta_loss = meta_loss + loss_q

        # 平均并做元更新
        meta_loss = meta_loss / len(tasks_batch)
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
