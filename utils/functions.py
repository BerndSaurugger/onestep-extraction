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

def best_scoring(filename="bb_attack_sdv1_30k_full_analyzed", amount=200, drop=False):
    
    df = pd.read_parquet(f'{filename}.parquet')
    df_sorted = df.sort_values(by='mse_real_gen', ascending=True)
    if drop:
        df_sorted.drop(columns=["edge_scores", "mse_real_gen", "overfit_type", "gen_seeds", "retrieved_urls", "template_indices"], inplace=True)
    df_sorted = df_sorted.head(amount)
    return df_sorted

def output_images(df_sorted, idx, attackfolder="gen_onestep/sdv1_bb_attack", best=True, output="onestep"):
    if best:
        row = df_sorted.iloc[idx]
    else:
        row = df_sorted.iloc[-idx-1]

    # Download the original image
    response = requests.get(row['url'])
    original_img = Image.open(BytesIO(response.content)).convert('RGB')

    pic = str(row['caption'])
    folder_name = prompt_to_folder(str(row['caption']), 200)
    base_folder = f"{attackfolder}/{folder_name}"

    # Load generated images
    gen_images = []
    for i in range(4):
        path = f"{base_folder}/000{i}.jpg"
        gen_img = Image.open(path).convert('RGB')
        gen_images.append(gen_img)

    # Combine original image and generated images horizontally
    all_images = [original_img] + gen_images
    widths, heights = zip(*(img.size for img in all_images))
    total_width = sum(widths)
    max_height = max(heights)

    combined_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in all_images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # Create output folder if it does not exist
    output_folder = 'output_combined_images' + f"/{output}"
    os.makedirs(output_folder, exist_ok=True)

    # Save combined image
    safe_filename = f"combined_{idx}.jpg"
    output_path = os.path.join(output_folder, safe_filename)
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