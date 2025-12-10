import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

import torch
import cv2

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for x1, y1, x2, y2, conf, cls in results.xyxy[0]:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[int(cls)]}{conf:.2f}",
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Real-Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import torch
# import cv2
# import sys

# print("=" * 50)
# print("🔍 DIAGNOSTICA YOLO + WEBCAM")
# print("=" * 50)

# # 1. Test OpenCV
# print("\n📹 Test 1: Verifica OpenCV")
# print(f"OpenCV version: {cv2.__version__}")

# # 2. Test accesso webcam
# print("\n📹 Test 2: Accesso webcam")
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ ERRORE: Impossibile aprire la webcam!")
#     print("Prova:")
#     print("  - cap = cv2.VideoCapture(1)  # prova indice diverso")
#     print("  - Controlla permessi webcam")
#     sys.exit(1)
# else:
#     print("✅ Webcam aperta correttamente")

# # Test lettura frame
# ret, test_frame = cap.read()
# if ret:
#     print(f"✅ Frame letto: {test_frame.shape}")
# else:
#     print("❌ Impossibile leggere frame dalla webcam")
#     cap.release()
#     sys.exit(1)

# # 3. Test caricamento modello
# print("\n🤖 Test 3: Caricamento modello YOLO")
# try:
#     print("Downloading/loading model... (può richiedere tempo la prima volta)")
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#     print("✅ Modello caricato con successo")
#     print(f"Device: {next(model.parameters()).device}")
# except Exception as e:
#     print(f"❌ ERRORE caricamento modello: {e}")
#     cap.release()
#     sys.exit(1)

# # 4. Test inferenza su singolo frame
# print("\n🔬 Test 4: Inferenza su frame di test")
# try:
#     frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
#     results = model(frame_rgb)
#     print(f"✅ Inferenza completata")
#     print(f"Oggetti rilevati: {len(results.xyxy[0])}")
#     if len(results.xyxy[0]) > 0:
#         print("Primi 3 detection:")
#         for i, det in enumerate(results.xyxy[0][:3]):
#             cls_id = int(det[5])
#             conf = float(det[4])
#             print(f"  {i+1}. {model.names[cls_id]} (conf: {conf:.2f})")
# except Exception as e:
#     print(f"❌ ERRORE durante inferenza: {e}")
#     cap.release()
#     sys.exit(1)

# # 5. Test finestra OpenCV
# print("\n🖼️ Test 5: Creazione finestra")
# try:
#     cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
#     cv2.imshow('Test Window', test_frame)
#     cv2.waitKey(1000)  # mostra per 1 secondo
#     cv2.destroyAllWindows()
#     print("✅ Finestra creata e chiusa correttamente")
# except Exception as e:
#     print(f"❌ ERRORE con finestra OpenCV: {e}")
#     print("Possibile problema con display/GUI")
#     cap.release()
#     sys.exit(1)

# print("\n" + "=" * 50)
# print("✅ TUTTI I TEST SUPERATI!")
# print("=" * 50)
# print("\n▶️  Avvio detection real-time...")
# print("Premi 'q' per uscire\n")

# # CODICE PRINCIPALE CON DEBUG
# frame_count = 0
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         print(f"⚠️ Frame {frame_count}: impossibile leggere")
#         break
    
#     frame_count += 1
    
#     # Feedback ogni 30 frames
#     if frame_count % 30 == 0:
#         print(f"📊 Frame processati: {frame_count}")
    
#     try:
#         # Inferenza
#         results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         # Disegna detection
#         detections = results.xyxy[0]
#         for x1, y1, x2, y2, conf, cls in detections:
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cls_id = int(cls)
            
#             # Rettangolo
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Etichetta
#             label = f"{model.names[cls_id]} {conf:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Info frame
#         cv2.putText(frame, f"Frame: {frame_count} | Objects: {len(detections)}",
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
#         # Mostra frame
#         cv2.imshow('YOLOv5 Real-Time Detection', frame)
        
#     except Exception as e:
#         print(f"❌ Errore frame {frame_count}: {e}")
#         break
    
#     # Esci con 'q'
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         print(f"\n👋 Uscita manuale dopo {frame_count} frames")
#         break

# print(f"\n📊 Statistiche finali:")
# print(f"  Frames totali processati: {frame_count}")
# print("  Chiusura risorse...")

# cap.release()
# cv2.destroyAllWindows()
# print("✅ Programma terminato correttamente")
