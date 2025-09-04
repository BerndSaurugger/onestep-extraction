import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import re

def prompt_to_folder(prompt, ml=200):
    # HTML-Tags entfernen
    sanitized = re.sub(r'<.*?>', '', prompt)
    # alles außer Buchstaben, Zahlen und _ durch _ ersetzen
    sanitized = re.sub(r'[^\w]', '_', sanitized)
    # mehrere _ zusammenfassen
    sanitized = re.sub(r'_+', '_', sanitized)
    # Länge begrenzen
    return sanitized[:ml]

def best_scoring(filename="bb_attack_sdv1_30k_full_analyzed", 
                 amount=200, 
                 drop=False, 
                 sort_by="edge_scores", 
                 ascending=False):
    """
    Lädt einen Parquet-DataFrame und gibt die Top-Einträge sortiert zurück.

    Args:
        filename (str): Parquet-Dateiname ohne ".parquet"
        amount (int): Anzahl der obersten Einträge, die zurückgegeben werden
        drop (bool): Ob bestimmte Spalten entfernt werden sollen
        sort_by (str): Name der Spalte, nach der sortiert werden soll
        ascending (bool): Sortierreihenfolge, True = aufsteigend, False = absteigend

    Returns:
        pd.DataFrame: Sortierter und ggf. beschnittener DataFrame
    """
    df = pd.read_parquet(f'{filename}.parquet')
    if sort_by not in df.columns:
        raise ValueError(f"Spalte '{sort_by}' existiert nicht im DataFrame.")
    
    df_sorted = df.sort_values(by=sort_by, ascending=ascending)
    
    if drop:
        cols_to_drop = ["edge_scores", "mse_real_gen", "overfit_type", 
                        "gen_seeds", "retrieved_urls", "template_indices"]
        df_sorted = df_sorted.drop(columns=[c for c in cols_to_drop if c in df_sorted.columns])
    
    return df_sorted.head(amount)

def safe_download_image(url, size=(256, 256)):
    """Versuche ein Bild von URL zu laden, ansonsten Platzhalter zurückgeben."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Fehler beim Herunterladen der URL {url}: {e}")
        return Image.new('RGB', size, color='gray')  # Platzhalter

def output_images(df_sorted, idx, attackfolder="gen_onestep/sdv1_bb_attack", best=True, output="onestep"):
    # Wähle Zeile
    row = df_sorted.iloc[idx] if best else df_sorted.iloc[-idx-1]

    # Lade Originalbild (robust)
    original_img = safe_download_image(row['url'])

    pic = str(row['caption'])
    folder_name = prompt_to_folder(pic, 200)
    base_folder = f"{attackfolder}/{folder_name}"

    # Lade generierte Bilder
    gen_images = []
    for i in range(10):
        path = f"{base_folder}/000{i}.jpg"
        try:
            gen_img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            gen_img = Image.new('RGB', (256, 256), color='gray')
        gen_images.append(gen_img)

    # Kombiniere Original + generierte Bilder horizontal
    all_images = [original_img] + gen_images
    widths, heights = zip(*(img.size for img in all_images))
    total_width = sum(widths)
    max_height = max(heights)

    combined_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in all_images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # Speichern
    output_folder = f'output_combined_images/{output}'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"combined_{idx}.jpg")
    combined_img.save(output_path)
    print(f"Saved combined image to {output_path}")

def prepare_for_multiple(input_file="bb_attack_sdv1_30k_full_analyzed.parquet", 
                         output_file="bb_attack_top400.parquet", 
                         amount=400, drop=False):
    # Original laden
    df = pd.read_parquet(input_file)

    # Kopie zum Sortieren
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='mse_real_gen', ascending=True)
    df_top = df_copy.head(amount)

    # IDs/Index der Top-Einträge extrahieren
    top_indices = df_top.index

    # Original auf Top-Einträge filtern, Reihenfolge bleibt erhalten
    df_filtered = df.loc[top_indices]

    if drop and "template_indices" in df_filtered.columns:
        df_filtered.drop(columns=["edge_scores", "mse_real_gen", "overfit_type", 
                                 "gen_seeds", "retrieved_urls", "template_indices"], inplace=True)

    # Ergebnis speichern
    df_filtered.to_parquet(output_file)
    print(f"Prepared data saved to {output_file}")