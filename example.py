import torch
from hrm import TransformerEmbeddingInput, TransformerRecurrentCell, TransformerOutputModule, HRM

# --- Configuration ---
VOCAB_SIZE = 5000       # Size of your vocabulary dictionary
LATENT_SIZE = 256       # Embedding dimension size
MAX_SEQ_LEN = 512        # Maximum length of your sequence
NUM_ATTN_HEADS = 8      # Number of attention heads for the transformers

# --- Instantiate the components ---
input_mod = TransformerEmbeddingInput(VOCAB_SIZE, LATENT_SIZE, MAX_SEQ_LEN)
l_mod = TransformerRecurrentCell(LATENT_SIZE, num_heads=NUM_ATTN_HEADS)
h_mod = TransformerRecurrentCell(LATENT_SIZE, num_heads=NUM_ATTN_HEADS)
output_mod = TransformerOutputModule(LATENT_SIZE, VOCAB_SIZE)

# --- Plug directly into your existing HRM container class ---
transformer_hrm = HRM(
    input_module=input_mod,
    l_module=l_mod,
    h_module=h_mod,
    output_module=output_mod
)

# --- Quick Functional Verification ---
if __name__ == "__main__":
    # Create fake sequence batch: (batch_size=4, seq_length=30)
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (4, 30))
    
    # Run a forward pass
    # N = 3 strategic planning loops, T = 5 quick worker loops
    predictions = transformer_hrm(dummy_tokens, num_outer_cycles=3, num_inner_steps=5)
    
    print(f"Input Token Shape:  {dummy_tokens.shape}")
    print(f"Output Logit Shape: {predictions.shape}") 
    # Output: torch.Size([4, 30, 5000]) -> (batch, sequence, vocab_logits)