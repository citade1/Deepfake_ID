import os
import pandas as pd 
from datasets import load_dataset, ClassLabel
from torch.utils.data import DataLoader

# from dotenv import load_dotenv
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(HF_TOKEN)
# dataset = load_dataset("HF_DATASET_REPO", token=True) ## load dataset from hugging face

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

dataset = ImageFolder(root="data", transform=transform)

file_paths, labels = zip(*dataset.samples)

dataset = train_test_split(
    file_paths, labels, test_size=0.1, shuffle=True, stratify=labels
)
data_train = dataset["train"]
data_test = dataset["test"]