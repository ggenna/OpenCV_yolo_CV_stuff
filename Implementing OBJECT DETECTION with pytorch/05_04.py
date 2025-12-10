import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import shutil
from tqdm import tqdm
from datetime import datetime

# ============================================
# SEZIONE 1: CLASSE DATASET YOLO CON VERBOSE
# ============================================

class WheatDatasetYOLO(Dataset):
    """
    Dataset per rilevamento teste di grano in formato YOLO.
    Converte bounding boxes da [x1,y1,x2,y2] a [class, x_center, y_center, width, height]
    """
    def __init__(self, annotations_file, img_dir, transform=None, verbose=True):
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔧 Inizializzazione WheatDatasetYOLO")
            print(f"{'='*60}")
            print(f"📄 File annotazioni: {annotations_file}")
            print(f"📁 Directory immagini: {img_dir}")
        
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = self.annotations['image_name'].unique()
        
        if self.verbose:
            print(f"✅ Annotazioni caricate: {len(self.annotations)} righe")
            print(f"✅ Immagini uniche trovate: {len(self.image_ids)}")
            print(f"{'='*60}\n")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Recupera tutte le boxes per questa immagine
        records = self.annotations[self.annotations['image_name'] == image_id]
        boxes = []
        
        for _, row in records.iterrows():
            box_str = row['BoxesString']
            if pd.isna(box_str) or box_str.strip().lower() == 'no_box':
                continue
            
            for b in box_str.split(';'):
                x1, y1, x2, y2 = map(float, b.strip().split())
                # Converti in formato YOLO (normalizzato 0-1)
                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                boxes.append([0, xc, yc, w, h])  # classe 0 = wheat

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes


# ============================================
# SEZIONE 2: CREAZIONE DATASET CON VERBOSE
# ============================================

def test_dataset(verbose=True):
    """Testa il dataset caricando il primo esempio"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"🧪 TEST DATASET")
        print(f"{'='*60}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = WheatDatasetYOLO(
        annotations_file='../gwhd_2021/competition_train.csv',
        img_dir='../gwhd_2021/images',
        transform=transform,
        verbose=verbose
    )

    if verbose:
        print(f"📊 Lunghezza dataset: {len(dataset)}")
        print(f"\n🔍 Caricamento primo esempio (idx=0)...")
    
    img, labels = dataset[0]
    
    if verbose:
        print(f"✅ Immagine caricata:")
        print(f"   - Shape: {img.shape} (C, H, W)")
        print(f"   - Tipo: {img.dtype}")
        print(f"   - Range valori: [{img.min():.3f}, {img.max():.3f}]")
        print(f"\n✅ Labels YOLO:")
        print(f"   - Shape: {labels.shape} (N_boxes, 5)")
        print(f"   - Numero boxes: {len(labels)}")
        print(f"   - Prime 3 boxes:")
        for i, box in enumerate(labels[:3]):
            print(f"     Box {i}: class={int(box[0])}, "
                  f"xc={box[1]:.3f}, yc={box[2]:.3f}, "
                  f"w={box[3]:.3f}, h={box[4]:.3f}")
        print(f"{'='*60}\n")
    
    return dataset


# ============================================
# SEZIONE 3: PREPARAZIONE DATASET YOLO
# ============================================

def convert_box(img_width, img_height, x1, y1, x2, y2):
    """Converte box da formato [x1,y1,x2,y2] a YOLO [class, xc, yc, w, h]"""
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [0, x_center, y_center, width, height]


def prepare_yolo_dataset(csv_path, image_dir, output_base='datasets/wheat', 
                        subset_ratio=0.25, val_split=0.1, verbose=True):
    """
    Prepara il dataset in formato YOLO con struttura di cartelle standard.
    
    Args:
        csv_path: Path al CSV con annotazioni
        image_dir: Directory con le immagini
        output_base: Directory di output
        subset_ratio: Percentuale del dataset da usare (0.25 = 25%)
        val_split: Percentuale per validazione (0.1 = 10%)
        verbose: Se True, stampa informazioni dettagliate
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 PREPARAZIONE DATASET YOLO")
        print(f"{'='*60}")
        print(f"⏰ Inizio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 CSV: {csv_path}")
        print(f"📁 Immagini: {image_dir}")
        print(f"💾 Output: {output_base}")
        print(f"📊 Subset ratio: {subset_ratio*100}%")
        print(f"📊 Validation split: {val_split*100}%")
    
    # Crea struttura cartelle
    os.makedirs(output_base, exist_ok=True)
    for sub in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        os.makedirs(os.path.join(output_base, sub), exist_ok=True)
    
    if verbose:
        print(f"✅ Struttura cartelle creata")
    
    # Carica annotazioni
    df = pd.read_csv(csv_path)
    all_image_ids = df['image_name'].unique()
    
    if verbose:
        print(f"\n📊 STATISTICHE DATASET:")
        print(f"   - Totale immagini: {len(all_image_ids)}")
        print(f"   - Totale annotazioni: {len(df)}")
    
    # Seleziona subset
    subset_size = int(subset_ratio * len(all_image_ids))
    image_ids = all_image_ids[:subset_size]
    
    # Split train/val
    val_count = int(len(image_ids) * val_split)
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])
    
    if verbose:
        print(f"\n📦 SUBSET SELEZIONATO:")
        print(f"   - Immagini totali: {len(image_ids)}")
        print(f"   - Training: {len(train_ids)}")
        print(f"   - Validation: {len(val_ids)}")
    
    # Contatori per statistiche
    stats = {
        'train': {'images': 0, 'boxes': 0, 'skipped': 0},
        'val': {'images': 0, 'boxes': 0, 'skipped': 0}
    }
    
    # Converti e copia files
    if verbose:
        print(f"\n🔄 Conversione in corso...")
    
    for img_id in tqdm(image_ids, desc="Conversione dataset", disable=not verbose):
        label_rows = df[df['image_name'] == img_id]
        img_path = os.path.join(image_dir, img_id)
        
        if not os.path.exists(img_path):
            if verbose and stats['train']['skipped'] + stats['val']['skipped'] < 5:
                print(f"⚠️  Immagine non trovata: {img_id}")
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            if verbose and stats['train']['skipped'] + stats['val']['skipped'] < 5:
                print(f"❌ Errore apertura {img_id}: {e}")
            continue

        # Converti boxes
        yolo_lines = []
        for _, row in label_rows.iterrows():
            if pd.isna(row['BoxesString']) or row['BoxesString'].strip().lower() == 'no_box':
                continue
            for box in row['BoxesString'].split(';'):
                try:
                    x1, y1, x2, y2 = map(float, box.strip().split())
                    yolo_box = convert_box(w, h, x1, y1, x2, y2)
                    yolo_lines.append(' '.join(map(str, yolo_box)))
                except:
                    continue

        # Determina subset
        subset = 'val' if img_id in val_ids else 'train'
        
        # Copia immagine
        shutil.copy(img_path, f"{output_base}/images/{subset}/{img_id}")
        
        # Scrivi label
        label_filename = img_id.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = f"{output_base}/labels/{subset}/{label_filename}"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Aggiorna statistiche
        stats[subset]['images'] += 1
        stats[subset]['boxes'] += len(yolo_lines)
        if len(yolo_lines) == 0:
            stats[subset]['skipped'] += 1
    
    # Stampa statistiche finali
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ CONVERSIONE COMPLETATA")
        print(f"{'='*60}")
        print(f"\n📊 STATISTICHE FINALI:")
        print(f"\n🏋️  TRAINING SET:")
        print(f"   - Immagini: {stats['train']['images']}")
        print(f"   - Bounding boxes: {stats['train']['boxes']}")
        print(f"   - Media boxes/immagine: {stats['train']['boxes']/max(stats['train']['images'],1):.2f}")
        print(f"   - Immagini senza boxes: {stats['train']['skipped']}")
        
        print(f"\n🔍 VALIDATION SET:")
        print(f"   - Immagini: {stats['val']['images']}")
        print(f"   - Bounding boxes: {stats['val']['boxes']}")
        print(f"   - Media boxes/immagine: {stats['val']['boxes']/max(stats['val']['images'],1):.2f}")
        print(f"   - Immagini senza boxes: {stats['val']['skipped']}")
        
        print(f"\n💾 Files salvati in: {output_base}")
        print(f"⏰ Fine: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
    
    return stats


# ============================================
# ESECUZIONE PRINCIPALE
# ============================================

if __name__ == "__main__":
    # Test del dataset
    dataset = test_dataset(verbose=True)
    
    # Preparazione dataset YOLO
    stats = prepare_yolo_dataset(
        csv_path='../gwhd_2021/competition_train.csv',
        image_dir='../gwhd_2021/images',
        output_base='datasets/wheat',
        subset_ratio=0.25,
        val_split=0.1,
        verbose=True
    )