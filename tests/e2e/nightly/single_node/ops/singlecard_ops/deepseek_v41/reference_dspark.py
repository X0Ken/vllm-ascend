def _apply_dsv4_rope(rotary_emb: nn.Module, positions: torch.Tensor, x: torch.Tensor, *, inverse: bool=False) -> torch.Tensor:
    (cos, sin) = get_cos_and_sin_dsa(positions)
    layer_name = rotary_emb.layername
    cos_t = cos[layer_name]
    sin_t = sin[layer_name]
    if inverse:
        sin_t = -sin_t
    return rotary_emb(x, cos_t, sin_t)

class DeepseekV4DSparkModel:

    def _project_shared_kv(self, hidden_states: torch.Tensor, positions: torch.Tensor, attn: type[nn.Module] | None=None) -> torch.Tensor:
        assert attn is not None
        kv = attn.kv_norm(attn.wkv(hidden_states))
        (k_nope, k_pe) = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()

    def precompute_and_store_context_kv(self, context_states: torch.Tensor, context_positions: torch.Tensor, context_slot_mapping: list[torch.Tensor | None] | None=None) -> None:
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        for (layer_idx, layer) in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            if context_positions.numel() == 0:
                return
            attn = layer.self_attn
            shared_kv = self._project_shared_kv(context_states, context_positions, attn)
            self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)
