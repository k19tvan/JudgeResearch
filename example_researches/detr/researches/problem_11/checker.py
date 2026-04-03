# learning_path/problem_11/checker.py
import torch
from starter import train_one_step

class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return {'pred_logits': self.w * x, 'pred_boxes': self.w * x}

class DummyMatcher:
    def forward(self, cost):
         return [(torch.tensor([0]), torch.tensor([0]))]

class DummyCrit(torch.nn.Module):
    def forward(self, out, tgt, idx):
         return {'loss_ce': out['pred_logits'].sum() * 0.1}

def check_loop():
    m = DummyNet()
    crit = DummyCrit()
    matcher = DummyMatcher()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    
    samples = torch.rand(1)
    
    try:
        val = train_one_step(m, crit, matcher, opt, samples, [])
        assert isinstance(val, float)
        assert m.w.grad is not None
        print("All Problem 11 checks passed")
    except NotImplementedError:
        print("Implement training loop iteration.")
        raise

if __name__ == "__main__":
    check_loop()
