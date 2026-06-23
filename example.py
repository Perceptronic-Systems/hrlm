import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from hrm import TransformerHRM


num_samples = 1000
SEQ_LEN = 1024
VOCAB_SIZE = 5000
LATENT_SIZE = 256   # Embedding dimension size
NUM_ATTN_HEADS = 8
BATCH_SIZE = 32
EPOCHS = 10
num_inner = 3
num_outer = 5

transformer_hrm = TransformerHRM(SEQ_LEN, VOCAB_SIZE, LATENT_SIZE, NUM_ATTN_HEADS)


X_data = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))
Y_data = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))

dataset = TensorDataset(X_data, Y_data)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.AdamW(transformer_hrm.parameters(), lr=1e-4, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

print(f"----- Beginning training on device {device} -----")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Clear gradients
        
        # Forward pass
        outputs = transformer_hrm(inputs, num_outer, num_inner)
        outputs_flat = outputs.view(-1, VOCAB_SIZE)
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat) # Calculate cross entropy loss
        loss.baackward() # Backward pass
        nn.utils.clip_grad_norm_(transformer_hrm.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {epoch_loss / len(train_loader):.4f}")

print("Training complete!")