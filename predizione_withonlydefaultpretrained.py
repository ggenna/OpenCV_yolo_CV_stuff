# pythorch for deep learning part 2
# Versione migliorata con filtraggio delle predizioni
# e visualizzazione delle etichette

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd

# Setup modello
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Carica immagine
image = Image.open("./maxresdefault.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

print("tensor shape",image_tensor.shape)

#print("image shape",image.shape)

# Inferenza
with torch.no_grad():
    predictions = model(image_tensor)

print("predictions:",predictions)

# Visualizza
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Soglia di confidenza minima per mostrare le predizioni
# Solo le predizioni con score >= threshold verranno visualizzate
threshold = 0.6

# Ottieni i nomi delle categorie COCO dal modello
# weights.meta["categories"] contiene la lista dei nomi delle classi
categories = weights.meta["categories"]

print("stampa categorie: ",categories)
# Itera su boxes, scores e labels insieme
# zip() combina le tre liste in modo da processarle simultaneamente
for box, score, label in zip(predictions[0]['boxes'], 
                               predictions[0]['scores'], 
                               predictions[0]['labels']):
    # Converti il punteggio in float Python per il confronto
    score_value = float(score)
    
    # Filtra le predizioni con bassa confidenza
    # Mostra solo le detection con score >= threshold
    if score_value >= threshold:
        # Converti coordinate in float Python
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # Disegna il rettangolo
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        # Aggiungi etichetta con nome classe e score
        # label è l'ID della classe (1-91 per COCO)
        # categories[label] restituisce il nome della classe
        label_text = f'{categories[label]}: {score_value:.2f}'
        
        # Posiziona il testo sopra la bounding box
        # bbox crea uno sfondo per rendere il testo leggibile
        ax.text(x1, y1 - 5, label_text,
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=2))

# Rimuovi gli assi per una visualizzazione più pulita
ax.axis('off')
plt.tight_layout()
plt.show()

#fig, ax = plt.subplots(1)
#ax.imshow(image)

for box in predictions[0]['boxes']:
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

# Salva la visualizzazione in un nuovo file PNG
# plt.savefig("output_con_bbox.png")   # crea "output_con_bbox.png" nella cartella corrente

# Puoi anche mostrare a schermo (opzionale)
# plt.show()
