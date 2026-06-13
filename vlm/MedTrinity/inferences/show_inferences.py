import json
import os
import textwrap
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
train_length = "55k"
images_base_dir = "D:/inferences/"  # Folder that contains the .png images

images_dir = os.path.join(images_base_dir, "shard_57")
output_dir = os.path.join(images_dir, f"MedTrinity25M_full_{train_length}")

jsonl_path = os.path.join(output_dir, "answers.jsonl")
output_folder = os.path.join(output_dir, "resultados_comparacion")

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

def wrap_text(text, width=65):
    """Formats long text into multiple lines for better readability."""
    if not text: return "N/A"
    return "\n".join(textwrap.wrap(text, width=width))

def main():
    # --- PROCESSING ---
    print(f"Loading data from {jsonl_path}...")
    all_cases = []
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found at {jsonl_path}")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_cases.append(json.loads(line))
                except:
                    continue

    # Select 10 random cases
    num_samples = min(50, len(all_cases))
    selected_cases = random.sample(all_cases, num_samples)
    print(f"Selected {num_samples} random images for comparison.")

    for data in tqdm(selected_cases, desc="Generating professional comparisons"):
        img_name = data["file_name"]
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path)

        # Professional layout: Narrower figure (16x10 instead of 20x12)
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.08)
        
        # --- LEFT SIDE: IMAGE ---
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f"{img_name}", fontsize=13, fontweight='bold', pad=10, color='#333333')
        ax_img.axis('off')

        # --- RIGHT SIDE: STRUCTURED TEXT ---
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')

        # Content setup (Narrower wrap to fit narrower column)
        source = data.get('source', 'N/A').upper()
        gt = wrap_text(data.get('caption', ''), width=55)
        base = wrap_text(data.get('model_base_response', ''), width=55)
        ft = wrap_text(data.get('model_ft_response', ''), width=55)

        # Build structured text with balanced spacing
        y_pos = 0.98
        
        def add_section(title, content, color, bold=False):
            nonlocal y_pos
            ax_text.text(0, y_pos, title, fontsize=12, fontweight='bold', color=color, transform=ax_text.transAxes)
            y_pos -= 0.025 # Gap between title and content
            if content:
                weight = 'bold' if bold else 'normal'
                ax_text.text(0, y_pos, content, fontsize=10.5, verticalalignment='top', 
                             family='monospace', transform=ax_text.transAxes, fontweight=weight)
                # Calculate dynamic height based on text lines
                num_lines = content.count('\n') + 1
                y_pos -= (num_lines * 0.021) + 0.06 # Increased base gap after section
            else:
                y_pos -= 0.02 

        add_section(f"FUENTE: {source}", "", "#555555")
        y_pos -= 0.02 # Separation after source
        
        add_section("--- REFERENCIA REAL (GROUND TRUTH) ---", gt, "#2c3e50")
        add_section("--- RESPUESTA MODELO BASE ---", base, "#c0392b")
        add_section("--- RESPUESTA MODELO AJUSTE FINO ---", ft, "#27ae60", bold=True)

        # 3. Save result (PNG + EPS)
        clean_name = os.path.splitext(img_name)[0]
        save_base = os.path.join(output_folder, f"{clean_name}_comparativa")
        
        plt.savefig(f"{save_base}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{save_base}.eps", format='eps', bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"\nProceso completado. Archivos .png y .eps guardados en: {output_folder}")

if __name__ == "__main__":
    main()
