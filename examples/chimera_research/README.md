# Chimera Research bridge

`chimera_research` is a compact MLA + shared-expert MoE architecture intended
for local experiments and HF checkpoint conversion.

The bridge reuses the maintained DeepSeek-V3 tensor layout, including MLA Q/KV
low-rank projections, routed experts, one shared expert, and expert weight
packing. Its deliberate research choices are four routed experts, top-1
routing, one initial dense layer, non-overlapped shared-expert execution, and a
standard auxiliary-loss router. The Transformers side additionally exposes an
optional Eagle proposal head and an entropy-regularized importance loss.

The Eagle head is returned only when requested (`return_eagle_logits=True`); it
is a proposal-logit interface, not a claim that a full acceptance scheduler is
implemented in Megatron.

Example:

```python
from transformers import ChimeraResearchConfig, ChimeraResearchForCausalLM
from megatron.bridge import AutoBridge

config = ChimeraResearchConfig()
model = ChimeraResearchForCausalLM(config)
bridge = AutoBridge.from_hf_config(config)
provider = bridge.to_megatron_provider(load_weights=False)
```
