from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider


@MegatronModelBridge.register_bridge(
    source="ChimeraResearchForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="chimera_research",
)
class ChimeraResearchBridge(DeepSeekV3Bridge):
    """Conversion bridge for the compact Chimera Research MLA/shared-MoE model.

    Weight layout intentionally follows DeepSeek-V3, so the inherited MLA,
    shared-expert, QKV, and expert mappings remain checkpoint-compatible.  The
    custom Eagle head is an HF-side proposal head and is not silently mapped into
    Megatron's verifier model; it can be exported separately when a speculative
    decoding runtime supports it.
    """

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = getattr(hf_pretrained.config, "router_aux_loss_coef", 1e-3)
        provider.moe_router_score_function = "softmax"
        provider.moe_router_enable_expert_bias = False
        provider.moe_shared_expert_overlap = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_layer_freq = [0] * hf_pretrained.config.first_k_dense_replace + [1] * (
            hf_pretrained.config.num_hidden_layers - hf_pretrained.config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = (
            hf_pretrained.config.moe_intermediate_size * hf_pretrained.config.n_shared_experts
        )
        provider.mtp_num_layers = None
        return provider
