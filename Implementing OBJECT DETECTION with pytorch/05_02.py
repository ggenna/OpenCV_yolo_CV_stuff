import torch 
print(f"PyTorch version: {torch.__version__}") 
print(f"Is CUDA available: {torch.cuda.is_available()}")


# modifica
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# utilizzeremo l'ottimizzatore Adam per mettere a punto il modello sul set di dati personalizzato
from torch.optim import Adam

print("load pre-trained Yolov5 model")
model = torch.hub.load('ultralytics/yolov5','yolov5s')



print("define customized class for dataset")
class Customdataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # load image and bounding box data here
        # apply trasformation if specified
        if self.transform:
            image = self.transform(image)
            target = []
        return image, target
    
# ora prepariamo il nostro set di dati personalizzato e utilizzare l'etichettatura
# Oppure usare ROboflow che crea etichette in formato Yolo
transform = transforms.Compose([transforms.ToTensor()])
dataset = Customdataset(
     annotations_file='../gwhd_2021/competition_train.csv',
        img_dir='../gwhd_2021/images',
        transform=transform,
)

#utilizzeremo l'ottimizzatore ADAM per metter a punto il modello sul set di dati personalizzato
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
optimizer = Adam(model.parameters(),lr=0.001)


#example training loop

for epoch in range(10):  
    print(f"\nEpoch {epoch}")
    for i, (images, targets) in data_loader:
        optimizer.zero_grad()
        loss = model(images, targets) # forward pass
        loss.backward()   #enable back propagation
        optimizer.step()  #update weights

        print(f"[Epoch {epoch} | Step {i}] Loss:")

# abbiamo caricato i pesi yolov5 preaaddestrati , preparato un set di dati personalizzato e addestrato il modello usando pytorch