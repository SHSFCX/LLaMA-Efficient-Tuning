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

if __name__ == "__main__":
    r = 4
    input_shape = (1, 2)
    input_embeds = torch.rand(input_shape)

    mask = prepare_r_attention_mask(r, input_shape, input_embeds)
    
    print(mask)