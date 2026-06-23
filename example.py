import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer
from hrm import TransformerHRM
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextCausalDataset(Dataset):
    def __init__(self, texts, tokenizer_name="bert-base-uncased", seq_len=512):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len
        
        # Adjust seq_len + 1 because we need context to shift targets
        # e.g., if we want seq_len=512, we need 513 tokens to get X (512) and Y (512)
        tokenized = self.tokenizer(
            texts,
            max_length=seq_len + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        self.input_ids = tokenized["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        token_chunk = self.input_ids[idx]
        
        # X: All tokens except the very last one
        x = token_chunk[:-1]
        # Y: All tokens except the very first one (shifted right by 1)
        y = token_chunk[1:]
        
        return x, y

raw_text_data = [
    "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers.",
    "Transformers are a type of neural network architecture that have become the baseline for modern natural language processing.",
    "Recurrent models process tokens sequentially, while attention mechanisms allow direct connection across long contexts."
] * 400


num_samples = 1000
SEQ_LEN = 512
LATENT_SIZE = 256   # Embedding dimension size
NUM_ATTN_HEADS = 8
BATCH_SIZE = 16
EPOCHS = 5
num_inner = 3
num_outer = 5
dataset = TextCausalDataset(raw_text_data, tokenizer_name="bert-base-uncased", seq_len=SEQ_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
VOCAB_SIZE = dataset.tokenizer.vocab_size

transformer_hrm = TransformerHRM(SEQ_LEN, VOCAB_SIZE, LATENT_SIZE, NUM_ATTN_HEADS)
transformer_hrm.to(device)


optimizer = optim.AdamW(transformer_hrm.parameters(), lr=1e-4, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

print(f"----- Beginning training on device {device} -----")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Clear gradients
        
        # Forward pass
        outputs = transformer_hrm(inputs, num_outer, num_inner)
        outputs_flat = outputs.view(-1, VOCAB_SIZE)
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat) # Calculate cross entropy loss
        loss.backward() # Backward pass
        nn.utils.clip_grad_norm_(transformer_hrm.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {epoch_loss / len(train_loader):.4f}")

print("Training complete!")

def generate_text(model, tokenizer, prompt, max_new_tokens=30, seq_len=512):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)  # Shape: [1, prompt_len]
    
    print(f"\n--- Generating text from prompt: \"{prompt}\" ---")
    
    # Disable gradient tracking for inference speed and memory safety
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop input if it exceeds the maximum sequence length the model supports
            if input_ids.size(1) > seq_len:
                input_ids_chunk = input_ids[:, -seq_len:]
            else:
                input_ids_chunk = input_ids
                
            outputs = model(input_ids_chunk, num_outer=5, num_inner=3)
            
            next_token_logits = outputs[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() in [tokenizer.pad_token_id, tokenizer.sep_token_id]:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


tokenizer = dataset.tokenizer

prompt_1 = "Deep learning is"
output_1 = generate_text(transformer_hrm, tokenizer, prompt_1, max_new_tokens=20, seq_len=SEQ_LEN)
print(f"Result: {output_1}")

prompt_2 = "Transformers are a type of"
output_2 = generate_text(transformer_hrm, tokenizer, prompt_2, max_new_tokens=20, seq_len=SEQ_LEN)
print(f"Result: {output_2}")