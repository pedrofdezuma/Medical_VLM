import matplotlib.pyplot as plt

# 1. Dataset setup for the final MRI dataset
categories = ['Control (Sano)', 'Esclerosis Múltiple', 'Epilepsia']
counts = [49470, 65275, 45396]
total = sum(counts)

# Format the total number using spaces as thousands separators
formatted_total = f"{total:,}".replace(",", " ")

# Generate legend labels with only category names
legend_labels = categories

# 2. Color palette configuration (Professional shades of blue, orange and green)
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Function to display both count and percentage inside slices
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # Format with space as thousands separator
        formatted_val = f"{val:,}".replace(",", " ")
        return f'{formatted_val}\n({pct:.1f}%)'
    return my_autopct

# 3. Create the figure
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('white')

# Plot the pie chart with internal labels
patches, texts, autotexts = ax.pie(
    counts,
    labels=None,
    colors=custom_colors,
    autopct=make_autopct(counts),
    startangle=140,
    pctdistance=0.75, # Distance from center to internal labels
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
)

# Style internal labels
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_weight('bold')
    autotext.set_fontsize(26)

# 4. Position the legend
ax.legend(
    patches, 
    legend_labels, 
    loc="center left", 
    bbox_to_anchor=(1.0, 0.5),
    fontsize=25, 
    frameon=True, 
    facecolor='white', 
    edgecolor='none'
)

# 5. Set the title
plt.title(f'Total de imágenes: {formatted_total}', 
          fontsize=35, fontweight='bold', pad=20)

# 6. Formatting and saving
plt.axis('equal')
plt.tight_layout()

# Save the figure to the TFG images directory
output_dir = "C:/Users/pedro/Desktop/TFG/images/"
output_name = 'distribucion_conjuntos_regresion'

# Ensure the directory exists (optional, but safer)
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, f'{output_name}.png'), dpi=300, facecolor='white', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{output_name}.eps'), format='eps', facecolor='white', bbox_inches='tight')

print(f"Gráfico guardado en: {output_dir}{output_name}.png y .eps")
print(f"Total procesado: {total} imágenes")

plt.show()
