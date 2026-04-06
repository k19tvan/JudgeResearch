# D-FINE Integration Tutorial

## 1. Final Assembled Project Tree
Once you have systematically solved problems `01` through `16`, you are ready to combine them into an end-to-end framework. Your ultimate directory structure will match the original scale, but rebuilt entirely from your tested pieces:

```
learning_path_project/
├── core/
│   ├── __init__.py
│   └── box_ops.py
├── nn/
│   ├── __init__.py
│   ├── hybrid_encoder.py
│   ├── dfine_decoder.py
│   ├── criterion.py
│   ├── matcher.py
│   ├── denoising.py
│   └── postprocessor.py
├── train.py
└── dfine.py
```

## 2. File Assembly Map

| Final File | Source Subproblem Files | What to copy | Why this location | Required imports/dependencies |
| :--- | :--- | :--- | :--- | :--- |
| `core/box_ops.py` | `01`, `02` | `box_cxcywh_to_xyxy`, `box_iou`, etc. | Pure math utility decoupled from models. | `torch` |
| `nn/criterion.py` | `03`, `13`, `14` | `sigmoid_focal_loss`, `SetCriterion`, Aux Losses | Loss logic centralized. | `nn`, `F`, `box_ops`, `matcher` |
| `nn/hybrid_encoder.py` | `04`, `05` | Projectors, BiFPN Feature Fusion Blocks | Extractor and intermediate scaler. | `nn`, `Conv2d`, `BatchNorm` |
| `nn/dfine_decoder.py` | `08`, `09`, `10` | Cross-Attention, Box Refiner, Decoupled Heads. | The defining logic of the D-FINE paper. | `nn`, `F`, `box_ops` |
| `nn/matcher.py` | `11`, `12` | Bipartite Matching Cost, Hungarian computation | Ground-truth setter logic. | `scipy.optimize`, `box_ops` |
| `nn/postprocessor.py` | `15` | `PostProcessor` block (top-k filter) | Cleans network outputs for inference. | `torch` |
| `dfine.py` | `16` | `End-to-End D-FINE Forward Pass` | The Capstone gluing Backbone, Encoder, Decoder. | `nn/*`, `core/*` |

## 3. Strict Merge Order
When assembling, you must merge the files in topological dependency order (no circular imports):
1. **Move `core/box_ops.py`** first. Test that it imports.
2. **Move `nn/matcher.py`** since it relies purely on box math.
3. **Move `nn/hybrid_encoder.py`** and `nn/postprocessor.py` (free-standing network blocks).
4. **Move `nn/dfine_decoder.py`** (relies on `box_ops.py`).
5. **Move `nn/criterion.py`** (relies on everything above to calculate total backward gradients).
6. **Move `dfine.py`** (The brain).

## 4. Common Errors and Fixes
- **Shape Mismatch**: Usually occurs in `BiFPN` concatenation across varying feature scales. Check that your multi-scale feature dimensionalities align closely before down/upsampling.
- **NaN/Inf Loss**: Check `sigmoid_focal_loss`. If logits diverge highly (e.g. -100 or +100), `torch.exp` logic internally might overflow. Ensure `inputs` are raw logits and you are using stable PyTorch `BCEWithLogitsLoss` natively.
- **Missing Batch Keys**: When tracking the `dict` out of the decoder, ensure you are always returning `pred_logits` and `pred_boxes`. Default dictionaries will crash `criterion.py` otherwise.
- **Device Mismatch (Expected `cuda:0`, got `cpu`)**: Always use `tensor.to(device)` or create auxiliary zero-tensors directly on the device using `torch.zeros(..., device=x.device)`.

## 5. Required Verification Commands
Inside `learning_path_project/`:

**Smoke Check (Instantiate cleanly):**
```bash
python -c "from dfine import EndToEndDFINE; model = EndToEndDFINE(); print(model)"
```

**Short-train check (10 iterations without data loader, random noise input):**
```bash
python -c "
import torch
from dfine import EndToEndDFINE
from nn.criterion import SetCriterion

model = EndToEndDFINE(); model.train()
crit = SetCriterion(...)

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

for _ in range(10):
    img = torch.rand(2, 3, 640, 640)
    out = model(img)
    loss = crit(out, fake_targets)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print('Loss:', loss.item())
"
```