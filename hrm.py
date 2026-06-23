import torch
import torch.nn as nn

class HRM(nn.Module):
    def __init__(self, input_module, h_module, l_module, output_module):
        super(HRM, self).__init__()
        self.input_module = input_module
        self.h_module = h_module 
        self.l_module = l_module 
        self.output_module = output_module

    def forward(self, x, num_outer_cycles=4, num_inner_steps=10):
        batch_size = x.size(0)
        
        latent_input = self.input_module(x)  # Now self will correctly refer to HRM
        
        h_state = torch.zeros_like(latent_input)
        l_state = torch.zeros_like(latent_input)

        for n in range(num_outer_cycles):
            for t in range(num_inner_steps):
                l_input = latent_input + h_state
                l_state = self.l_module(l_input, l_state)

            h_state = self.h_module(l_state, h_state)
            l_state = torch.tanh(h_state)

        output = self.output_module(l_state)
        return output

class TransformerEmbeddingInput(nn.Module):
    def __init__(self, vocab_size, latent_size, max_seq_len=100):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, latent_size)
        self.pos_emb = nn.Embedding(max_seq_len, latent_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)

# 2. H & L Modules: Wrapping a Transformer Layer to act like a recurrent Cell
class TransformerRecurrentCell(nn.Module):
    def __init__(self, latent_size, num_heads=4):
        super().__init__()
        # We use a standard encoder layer acting as our internal processing step
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=latent_size, 
            nhead=num_heads, 
            dim_feedforward=latent_size * 2,
            batch_first=True,
            norm_first=True
        )
        
    def forward(self, current_input, hidden_state):
        fused_state = current_input + hidden_state
        new_hidden_state = self.transformer_layer(fused_state)
        return new_hidden_state

# 3. Output Module: Projects latent sequence back to vocabulary space
class TransformerOutputModule(nn.Module):
    def __init__(self, latent_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(latent_size, vocab_size)
        
    def forward(self, latent_state):
        # Project back to vocabulary logit space
        return self.linear(latent_state)

class TransformerHRM(nn.Module):
    def __init__(self, vocab_isze, seq_len, vocab_size, latent_size, num_attn_heads):
        super().__init__()
        self.input_mod = TransformerEmbeddingInput(vocab_size, latent_size, seq_len)
        self.l_mod = TransformerRecurrentCell(latent_size, num_heads=num_attn_heads)
        self.h_mod = TransformerRecurrentCell(latent_size, num_heads=num_attn_heads)
        self.output_mod = TransformerOutputModule(latent_size, vocab_size)

        self.transformer_hrm = HRM(
            input_module=input_mod,
            l_module=l_mod,
            h_module=h_mod,
            output_module=output_mod
        )

    def forward(self, x, num_outer, num_inner):
        return transformer_hrm(x, num_outer_cycles=num_outer, num_inner_steps=num_inner)