from torch.utils.data import DataLoader
from dataset import Dataset_Train
from dataset import Dataset_Test
import segmentation_models_pytorch as smp
import torch
from model import UNet_Real
from timeit import default_timer as timer
import tqdm
import torchmetrics
from torchmetrics.segmentation import DiceScore
import torchvision.transforms as T
import os

train_dataset = Dataset_Train(True)
test_dataset = Dataset_Test(True)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = UNet_Real().to(device)
loss_fn = smp.losses.DiceLoss('binary').to(device)
optimizer1 = torch.optim.Adam(model_1.parameters(), lr=0.001)

num_epochs = 2
dice_score_fn = DiceScore(num_classes=1, average='none').to(device)

def train_loop(dataloader, model, loss_fn, loss, optimizer, dice_score):
  for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = (target[:, 0, :, :] > 0).long().unsqueeze(1)

        loss = loss_fn(output, target)
        dice_score = dice_score_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item()
        dice_score += dice_score.item()
  print(f"Train Loss: {loss/len(train_dataloader)}")
  print(f"Test Dice Score: {dice_score/len(test_dataloader)}")

def test_loop(dataloader, model, loss_fn, loss, dice_score):
  with torch.inference_mode():
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = (target[:, 0, :, :] > 0).long().unsqueeze(1)

        loss = loss_fn(output, target)
        dice_score = dice_score_fn(output, target)

        loss += loss.item()
        dice_score += dice_score.item()
    print(f"Test Loss: {loss/len(test_dataloader)}")
    print(f"Test Dice Score: {dice_score/len(test_dataloader)}")

start_time = timer()

for epoch in tqdm(range(num_epochs)):
    print(f"EPOCH: {epoch}\n-------")
    model_1.train();
    loss = 0.0
    dice_score = 0.0
    train_loop(train_dataloader, model_1, loss_fn, loss, optimizer1, dice_score)

    model_1.eval()
    val_loss = 0.0
    val_dice_score = 0.0

    test_loop(test_dataloader, model_1, loss_fn, val_loss, val_dice_score)

end_time = timer()
print(f"Total time taken: {end_time - start_time}")

output_dir = 'predictions'

with torch.inference_mode():
    i = 0
    for batch_idx, (data, target) in enumerate(test_dataloader):
        if i >= 1:
          break
        else:
          data, target = data.to(device), target.to(device)
          output = model_1(data)
          target = (target[:, 0, :, :] > 0).float().unsqueeze(1)

          transform = T.ToPILImage()

          print(data.shape)
          print(output.shape)
          print(target.shape)

          i += 1

def save_images(input_tensor):
  with torch.no_grad():
    for i, image_tensor in enumerate(input_tensor):
      image = transform(image_tensor)
      image.save(os.path.join(output_dir, f'image_mask{i}.png'))

save_images(output)