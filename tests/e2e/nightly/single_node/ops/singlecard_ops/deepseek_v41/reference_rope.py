class RopeGlobalState:

    def __init__(self):
        self.full_rope_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.runtime_buffer: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.spec_runtime_buffer: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.layer_info: dict[str, tuple[str, list[str]]] = {}
        self.registry_summary: dict[str, set] = {}

class RopeDataProxy:

    def __init__(self, data_map, is_cos=True):
        self._data = data_map
        self.idx = 0 if is_cos else 1

    def pad_to(self, target_len: int, dim: int=0) -> 'RopeDataProxy':
        """
        Return a new proxy whose underlying tensors are padded to ``target_len`` along ``dim``.
        """
        new_data: dict = {}
        for (config_key, groups) in self._data.items():
            new_data[config_key] = {}
            for (group_name, (cos_t, sin_t)) in groups.items():
                pad_size = target_len - cos_t.shape[dim]
                if pad_size > 0:
                    ndim = cos_t.ndim
                    pad = [0] * (2 * ndim)
                    pad[-(1 + 2 * dim)] = pad_size
                    cos_t = F.pad(cos_t, pad)
                    sin_t = F.pad(sin_t, pad)
                new_data[config_key][group_name] = (cos_t, sin_t)
        return RopeDataProxy(new_data, is_cos=self.idx == 0)

    def __getitem__(self, index):
        if not isinstance(index, str):
            new_map: dict = {}
            for (config_k, groups_map) in self._data.items():
                new_map[config_k] = {}
                for (group_name, item) in groups_map.items():
                    c_val = item[0][index]
                    s_val = item[1][index]
                    new_map[config_k][group_name] = (c_val, s_val)
            return RopeDataProxy(new_map, is_cos=self.idx == 0)
        else:
            layername = index
            info = _ROPE_STATE.layer_info.get(layername)
            if info is None:
                raise KeyError(f'Layer {layername} not registered.')
            (config_key, required_groups) = info
            config_data = self._data.get(config_key, {})
            layer_result = {}
            for grp in required_groups:
                if grp in config_data:
                    layer_result[grp] = config_data[grp][self.idx]
                else:
                    pass
            if len(layer_result) == 1:
                return list(layer_result.values())[0]
            return layer_result

def get_cos_and_sin_dsa(positions: torch.Tensor | dict[str, torch.Tensor], use_cache: bool=False, draft_index: int | None=None):
    if isinstance(positions, torch.Tensor):
        pos_map = {'default': positions}
    else:
        pos_map = positions
    batch_result: dict[Any, Any] = {}
    for (config_key, registered_groups) in _ROPE_STATE.registry_summary.items():
        if config_key not in _ROPE_STATE.full_rope_cache:
            continue
        (full_rope_cos, full_rope_sin) = _ROPE_STATE.full_rope_cache[config_key]
        batch_result[config_key] = {}
        for (group_name, pos_tensor) in pos_map.items():
            if group_name not in registered_groups:
                continue
            if use_cache:
                group_buffers = _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name) if draft_index is None else _ROPE_STATE.spec_runtime_buffer.get(config_key, {}).get(group_name)
                if group_buffers is None:
                    continue
                (buf_cos, buf_sin) = group_buffers
                num_tokens = pos_tensor.size(0)
                gather_idx = pos_tensor.to(torch.long).reshape(-1, 1, 1, 1).expand(num_tokens, 1, 1, full_rope_cos.size(-1))
                if draft_index is None:
                    torch.gather(full_rope_cos, 0, gather_idx, out=buf_cos[:num_tokens])
                    torch.gather(full_rope_sin, 0, gather_idx, out=buf_sin[:num_tokens])
                    batch_result[config_key][group_name] = (buf_cos[:num_tokens], buf_sin[:num_tokens])
                else:
                    torch.gather(full_rope_cos, 0, gather_idx, out=buf_cos[draft_index - 1][:num_tokens])
                    torch.gather(full_rope_sin, 0, gather_idx, out=buf_sin[draft_index - 1][:num_tokens])
                    batch_result[config_key][group_name] = (buf_cos[draft_index - 1][:num_tokens], buf_sin[draft_index - 1][:num_tokens])
            else:
                curr_cos = full_rope_cos[pos_tensor]
                curr_sin = full_rope_sin[pos_tensor]
                batch_result[config_key][group_name] = (curr_cos, curr_sin)
    return (RopeDataProxy(batch_result, is_cos=True), RopeDataProxy(batch_result, is_cos=False))

class ComplexExpRotaryEmbedding:

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ori_shape = x.shape
        y = x
        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='interleave')
        y.copy_(x.view(ori_shape))
        return y
