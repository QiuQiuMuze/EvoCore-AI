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

        meta_loss = torch.tensor(0., device=self.device)

        for task in tasks_batch:
            support = task['support']  # List of (state_tensor[C,H,W], action, label)
            query   = task['query']

            inner_opt = optim.SGD(self.processor.parameters(), lr=self.lr_inner)
            with innerloop_ctx(self.processor, inner_opt, device=self.device) as (fmodel, diffopt):
                # —— inner‐loop on support —— #
                # 1) 把原始状态列表载到 device 上
                s_states = [t[0].to(self.device) for t in support]  # each: (C,H,W)
                s_labels = torch.tensor([t[2] for t in support],
                                        dtype=torch.long, device=self.device)  # (K,)

                # 2) 逐样本抽特征，再拼 batch
                feat_s_list = [fmodel._extract_features(s) for s in s_states]
                # 如果你的 _extract_features 恰好返回 (d_model,)，那上面列表项就是一维向量。
                # 如果它返回 (L,d_model)，你可以在这里做 flatten：
                # feat_s_list = [feat.view(-1) for feat in feat_s_list]
                feat_s = torch.stack(feat_s_list, dim=0)  # → (K, d_model)

                logits_s = fmodel.classifier(feat_s)  # → (K, num_classes)
                loss_s = F.cross_entropy(logits_s, s_labels)
                diffopt.step(loss_s)

                # —— query phase —— #
                q_states = [t[0].to(self.device) for t in query]  # each: (C,H,W)
                q_labels = torch.tensor([t[2] for t in query],
                                        dtype=torch.long, device=self.device)  # (K',)

                feat_q_list = [fmodel._extract_features(q) for q in q_states]
                # 同样如有需要 flatten：
                # feat_q_list = [fq.view(-1) for fq in feat_q_list]
                feat_q = torch.stack(feat_q_list, dim=0)  # → (K', d_model)

                logits_q = fmodel.classifier(feat_q)  # → (K', num_classes)
                loss_q = F.cross_entropy(logits_q, q_labels)
                meta_loss = meta_loss + loss_q

        # 平均并做一次 meta 更新
        meta_loss = meta_loss / len(tasks_batch)
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()

