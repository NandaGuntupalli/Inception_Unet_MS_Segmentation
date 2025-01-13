from pathlib import Path
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage import io

def find_patient(image_path):
  image_path = str(image_path)
  parts = image_path.split("/")
  patient = parts[5]
  return patient

def find_section(image_path):
  image_path = str(image_path)
  parts = image_path.split("/")
  section = parts[4]
  section = section + "annot"
  return section

def find_slice_train(image_path):
  image_path = str(image_path)
  filename = image_path.split("/")[-1]
  parts = filename.split("_")
  last_part = parts[-1]
  numbers = ''.join(filter(str.isdigit, last_part))
  return numbers

def find_slice_test(image_path):
  image_path = str(image_path)
  filename = image_path.split("/")[-1]
  parts = filename.split("_")
  last_part = parts[-1]
  numbers = ''.join(filter(str.isdigit, last_part))
  numbers = numbers[1:]
  return numbers

def patient_to_path(patient):
  path = patient.replace("P", "0")
  path = path.replace("T", "0")
  return path

def patient_to_path_test(image_path):
  if find_section(image_path) == "test1annot":
    return "01_04"
  if find_section(image_path) == "test2annot":
    return "02_04"
  if find_section(image_path) == "test3annot":
    return "03_04"
  if find_section(image_path) == "test4annot":
    return "04_04"
  if find_section(image_path) == "test5annot":
    return "05_04"
  return None

data_transform = transforms.Compose([
  transforms.Resize((128, 128)),
  # Flip the image horizontally
  transforms.RandomHorizontalFlip(p=0.5),
  # Turns the image into a tensor
  transforms.ToTensor()
])

data_path = Path("data/")

np.set_printoptions(threshold=3)

Train_X = Path("data/ISBI_2015/Train/X")
Train_Y = Path("data/ISBI_2015/Train/Y")

Test_X = Path("data/ISBI_2015/Test/X")
Test_Y = Path("data/ISBI_2015/Test/Y")

class Dataset_Train():
  def __init__(self, augmentation=None):
    self.image_paths = list(Train_X.glob("**/*/*.png"))
    self.augmentation = augmentation

  def __getitem__(self, i):
    image_path = self.image_paths[i]

    image = io.imread(image_path)
    section = find_section(image_path)
    patient = find_patient(image_path)
    scan = find_slice_train(image_path)
    mask_path = "data/ISBI_2015_Small/Train/Y/"+section+"/"+patient+"/training"+patient_to_path(patient)+"_mask1"+scan+".png"
    mask = io.imread(mask_path)
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)


    t_image = data_transform(image)
    t_imagemask = data_transform(mask)

    return t_image, t_imagemask

  def __len__(self):
    return len(self.image_paths)

class Dataset_Test():
  def __init__(self, augmentation):
    self.image_paths = list(Test_X.glob("**/*/*.png"))
    self.augmentation = augmentation

  def __getitem__(self, i):
    image_path = self.image_paths[i]

    image = io.imread(image_path)
    section = find_section(image_path)
    scan = find_slice_train(image_path)
    patient = patient_to_path_test(image_path)
    mask_path = "data/ISBI_2015_Small/Test/Y/"+section+"/Patient-1/training"+patient+"_mask1"+scan+".png"
    mask = io.imread(mask_path)
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    t_image = data_transform(image)
    t_imagemask = data_transform(mask)

    return t_image, t_imagemask

  def __len__(self):
    return len(self.image_paths)