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

# STAMPIAMO LA STRUTTURA DEL MODELLO
# Questo ci mostrerà tutti i "figli" diretti di model.model
# Stiamo cercando il nome della lista nn.Sequential
    # print("--- INIZIO STRUTTURA MODELLO ---")
    # print(model.model)
    # print("--- FINE STRUTTURA MODELLO ---")

#freeze initial layers
for param in model.parameters():
    param.requires_grad = False


# unfreeze deeper layers
print("Unfreezing the detection head...")
for param in model.model.model.model[-2].parameters():  # Questo è il modulo 'Detect'
    param.requires_grad = True 

print("Model setup complete. Ready for fine-tuning.")

# Ora puoi procedere con la definizione del tuo Adam optimizer
# Nota: devi passare solo i parametri che richiedono un gradiente!

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0001)

#example training loop

for epoch in range(5):  #fewer epoch for fine tuning
    print(f"\nEpoch {epoch}")
    for i, (images, targets) in data_loader:
        optimizer.zero_grad()
        loss = model(images, targets) # forward pass
        loss.backward()   #enable back propagation
        optimizer.step()  #update weights

        print(f"[Epoch {epoch} | Step {i}] Loss:")



# misuring performance
model.eval()

with torch.no_grad():
    for i, (images, targets) in val_loader:
        predictions = model(images) # forward pass
        iou = calculate_iou(predictions['boxes'],targets['boxes'])
        map_score = calculate_map(predictions['boxes'],targets['boxes'])

    print(f"fine tuning iou {iou} | fine tuning map {map_score}] ")