from higher import innerloop_ctx
import torch
class MetaTrainer:
    def __init__(self, processor, lr_inner=1e-3, lr_meta=1e-4):
        self.processor = processor
        self.meta_opt = torch.optim.Adam(processor.parameters(), lr=lr_meta)
        self.lr_inner = lr_inner

    def meta_update(self, tasks_batch):
        """
        tasks_batch: List of tasks; each task 是 (support_set, query_set)
        support_set: [(state, label), ...]
        query_set: 同格式
        """
        if not tasks_batch:  # ✅ 空检查
            return
        meta_loss = 0.0
        for support, query in tasks_batch:
            with innerloop_ctx(self.processor, torch.optim.SGD, {'lr':self.lr_inner}) as (fmodel, diffopt):
                # inner-loop 训练
                for state, label in support:
                    pred = fmodel._extract_features(state)
                    l = torch.nn.functional.cross_entropy(pred.unsqueeze(0), torch.tensor([label]))
                    diffopt.step(l)
                # query 上评估
                for state, label in query:
                    pred_q = fmodel._extract_features(state)
                    meta_loss += torch.nn.functional.cross_entropy(pred_q.unsqueeze(0), torch.tensor([label]))
        meta_loss /= len(tasks_batch)
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
