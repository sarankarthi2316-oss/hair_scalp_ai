import matplotlib.pyplot as plt
import numpy as np

# High-resolution styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Benchmark Data
models = [
    'Logistic\nRegression',
    'Decision\nTree',
    'Random\nForest',
    'Support Vector\nMachine',
    'Gradient\nBoosting',
]
accuracy = [84.00, 81.33, 91.33, 86.67, 94.67]
f1_score = [84.30, 81.50, 91.50, 86.70, 94.85]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

rects1 = ax.bar(
    x - width / 2,
    accuracy,
    width,
    label='Accuracy (%)',
    color='#0284c7',
    edgecolor='#0f172a',
)
rects2 = ax.bar(
    x + width / 2,
    f1_score,
    width,
    label='F1-Score (%)',
    color='#10b981',
    edgecolor='#0f172a',
)

ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax.set_title(
    'Performance Comparison of Candidate Classification Models',
    fontsize=13,
    fontweight='bold',
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_ylim(70, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)


# Add percentage labels above bars
def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate(
        f'{height:.1f}%',
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('fig_6_3_model_accuracy.png')
plt.close()
print('✓ Saved: fig_6_3_model_accuracy.png')