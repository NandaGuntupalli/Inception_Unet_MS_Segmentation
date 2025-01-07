from torch.utils.data import DataLoader
from dataset import Dataset_Train
from dataset import Dataset_Test
import segmentation_models_pytorch as smp
import torch
from model import UNet_Real
from timeit import default_timer as timer
import tqdm

train_dataset = Dataset_Train(True)
test_dataset = Dataset_Test(True)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = UNet_Real().to(device)
loss_fn = smp.losses.DiceLoss('binary').to(device)
optimizer1 = torch.optim.Adam(model_1.parameters(), lr=0.001)

num_epochs = 1

def train_loop(dataloader, model, loss_fn, loss, optimizer):
  for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = (target[:, 0, :, :] > 0).float().unsqueeze(1)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss += loss.item()
  print(f"Train Loss: {loss/len(train_dataloader)}")

def test_loop(dataloader, model, loss_fn, loss):
  with torch.inference_mode():
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = (target[:, 0, :, :] > 0).float().unsqueeze(1)
        loss = loss_fn(output, target)
        loss += loss.item()
    print(f"Test Loss: {loss/len(test_dataloader)}")

start_time = timer()

for epoch in tqdm(range(num_epochs)):
    print(f"EPOCH: {epoch}\n-------")
    model_1.train();
    loss = 0.0
    train_loop(train_dataloader, model_1, loss_fn, loss, optimizer1)

    model_1.eval()
    val_loss = 0.0

    test_loop(test_dataloader, model_1, loss_fn, val_loss)

end_time = timer()
print(f"Total time taken: {end_time - start_time}")