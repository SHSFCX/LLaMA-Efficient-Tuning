import torch

def prepare_r_attention_mask(r, input_shape, inputs_embeds, past_key_values_length=None):
    # create causal mask
    # input_shape: B x r, L/r
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    seq_len = r * input_shape[-1]

    
    mask_self = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask_self.size(-1), device=device)
    mask_self.masked_fill_(mask_cond < (mask_cond + 1).view(mask_self.size(-1), 1), 0)

    mask_cond = torch.arange(0, r, device=device)
    mask_cond = mask_cond.view(-1, 1).expand(-1, input_shape[-1]).reshape(-1)
    mask_self.masked_fill_(mask_cond < mask_cond.view(-1, 1), torch.finfo(dtype).min)
    mask_self = mask_self.to(dtype)

    mask_ref = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(0, r, device=device)
    mask_cond = mask_cond.view(-1, 1).expand(-1, input_shape[-1]).reshape(-1)
    mask_ref.masked_fill_(mask_cond < mask_cond.view(-1, 1), 0)
    mask_ref = mask_ref.to(dtype)

    attention_mask = torch.cat((mask_self, mask_ref), dim=-1)

    return attention_mask


def prepare_r_attention_mask_2(ref_attn_mask, iqa_s_attention_mask, batch_size, seq_length, instruct_length, ref_length, num_ref):
    # create causal mask
    # input_shape: B x r, L/r
    dtype = ref_attn_mask.dtype
    device = ref_attn_mask.device

    left_mask_up = torch.full([instruct_length, ref_length * num_ref], torch.finfo(dtype).min, device=device)
    left_mask_up = left_mask_up[None, None, :, :].expand(batch_size, 1, -1, -1).clone()
    left_mask_down = torch.full(
        [seq_length - instruct_length - ref_length * num_ref, ref_length * num_ref],
        0,
        device=device
    )
    left_mask_down = left_mask_down[None, None, :, :].expand(batch_size, 1, -1, -1).clone()
    left_mask = torch.cat([left_mask_up, left_mask_down], dim=-2)
    
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_mask = ref_attn_mask[:, None, None, :].expand(
        batch_size, 1, seq_length - ref_length * num_ref, ref_length * num_ref).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    expanded_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
    left_mask = (left_mask + expanded_mask).to(dtype)

    r_attention_mask = torch.cat([left_mask, iqa_s_attention_mask], dim=-1)

    return r_attention_mask


if __name__ == "__main__":
    # r = 4
    # input_shape = (1, 2)
    # input_embeds = torch.rand(input_shape)

    # mask = prepare_r_attention_mask(r, input_shape, input_embeds)
    iqa_s_attention_mask = [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]
    iqa_s_attention_mask = torch.tensor(iqa_s_attention_mask, dtype=torch.float16)
    iqa_s_attention_mask = iqa_s_attention_mask[None, None, :, :].expand(1, 1, -1, -1)

    inverted_mask = 1.0 - iqa_s_attention_mask
    iqa_s_attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(torch.float16).min)

    ref_attn_mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.float16)

    mask = prepare_r_attention_mask_2(
        ref_attn_mask,
        iqa_s_attention_mask,
        batch_size=1,
        seq_length=8,
        instruct_length=2,
        ref_length=2,
        num_ref=2,
    )
    
    print(mask)