import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# ==========================================================
# FIG 5.1: Dashboard User Input Form
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_facecolor('#090d16')
fig.patch.set_facecolor('#090d16')

# Title
plt.text(
    0.5,
    0.93,
    'FollicleAI Diagnostics — User Input Interface',
    color='#38bdf8',
    fontsize=14,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.5,
    0.88,
    'Multimodal Image Capture & Clinical Questionnaire Module',
    color='#94a3b8',
    fontsize=10,
    ha='center',
)

# Upload Box 1 (Scalp)
box1 = patches.FancyBboxPatch(
    (0.08, 0.45),
    0.4,
    0.38,
    boxstyle='round,pad=0.02',
    edgecolor='#38bdf8',
    facecolor='#1e293b',
    linewidth=1.5,
)
ax.add_patch(box1)
plt.text(0.28, 0.72, '🔬', fontsize=26, ha='center')
plt.text(
    0.28,
    0.63,
    'Scalp Close-Up Image',
    color='#ffffff',
    fontsize=11,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.28,
    0.57,
    'Status: [✓ scalp_macro_sample.jpg]',
    color='#34d399',
    fontsize=9,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.28,
    0.50,
    'Evaluates 14 Scalp Pathologies',
    color='#94a3b8',
    fontsize=8.5,
    ha='center',
)

# Upload Box 2 (Head)
box2 = patches.FancyBboxPatch(
    (0.52, 0.45),
    0.4,
    0.38,
    boxstyle='round,pad=0.02',
    edgecolor='#38bdf8',
    facecolor='#1e293b',
    linewidth=1.5,
)
ax.add_patch(box2)
plt.text(0.72, 0.72, '👤', fontsize=26, ha='center')
plt.text(
    0.72,
    0.63,
    'Head / Crown Photo',
    color='#ffffff',
    fontsize=11,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.72,
    0.57,
    'Status: [✓ head_crown_sample.jpg]',
    color='#34d399',
    fontsize=9,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.72,
    0.50,
    'Determines Norwood Density Scale',
    color='#94a3b8',
    fontsize=8.5,
    ha='center',
)

# Questionnaire Summary Area
q_box = patches.FancyBboxPatch(
    (0.08, 0.12),
    0.84,
    0.28,
    boxstyle='round,pad=0.02',
    edgecolor='#334155',
    facecolor='#1e293b',
    linewidth=1.2,
)
ax.add_patch(q_box)
plt.text(
    0.11,
    0.33,
    'Clinical Biomarkers & Lifestyle Input Configuration:',
    color='#cbd5e1',
    fontsize=10,
    fontweight='bold',
)
plt.text(
    0.11,
    0.26,
    '• Patient Age: 28 yrs             • Genetics: Family History (Yes)         '
    '  • Stress Level: High',
    color='#94a3b8',
    fontsize=9,
)
plt.text(
    0.11,
    0.20,
    '• Medical Condition: Alopecia     • Deficiencies: Zinc, Biotin Deficiency  '
    '  • Hormones: Active Shift',
    color='#94a3b8',
    fontsize=9,
)
plt.text(
    0.11,
    0.14,
    '• Smoking Status: Smoker         • Weight Loss: Stable (<2kg)             '
    ' • Habits: Poor Care',
    color='#94a3b8',
    fontsize=9,
)

# Button
btn = patches.FancyBboxPatch(
    (0.3, 0.02),
    0.4,
    0.065,
    boxstyle='round,pad=0.01',
    edgecolor='#0284c7',
    facecolor='#0284c7',
)
ax.add_patch(btn)
plt.text(
    0.5,
    0.045,
    '⚡ RUN MULTIMODAL AI DIAGNOSTICS',
    color='#ffffff',
    fontsize=10,
    fontweight='bold',
    ha='center',
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('fig_5_1_dashboard_input.png')
plt.close()
print('✓ Saved: fig_5_1_dashboard_input.png')

# ==========================================================
# FIG 5.2: Dashboard Health Score, Risk Level & Factor Analysis
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_facecolor('#090d16')
fig.patch.set_facecolor('#090d16')

plt.text(
    0.5,
    0.93,
    'Diagnostic Summary & Factor Analysis Dashboard',
    color='#38bdf8',
    fontsize=14,
    fontweight='bold',
    ha='center',
)

# Card 1: Scalp Condition
c1 = patches.FancyBboxPatch(
    (0.06, 0.58),
    0.26,
    0.28,
    boxstyle='round,pad=0.02',
    edgecolor='#334155',
    facecolor='#1e293b',
)
ax.add_patch(c1)
plt.text(
    0.19,
    0.80,
    'SCALP CONDITION',
    color='#94a3b8',
    fontsize=9,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.19,
    0.72,
    'ALOPECIA AREATA',
    color='#38bdf8',
    fontsize=12,
    fontweight='bold',
    ha='center',
)
plt.text(0.19, 0.64, 'Confidence: 89.20%', color='#34d399', fontsize=10, ha='center')

# Card 2: Density
c2 = patches.FancyBboxPatch(
    (0.37, 0.58),
    0.26,
    0.28,
    boxstyle='round,pad=0.02',
    edgecolor='#334155',
    facecolor='#1e293b',
)
ax.add_patch(c2)
plt.text(
    0.50,
    0.80,
    'NORWOOD DENSITY',
    color='#94a3b8',
    fontsize=9,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.50,
    0.72,
    'TYPE 3 (EARLY THINNING)',
    color='#2dd4bf',
    fontsize=11,
    fontweight='bold',
    ha='center',
)
plt.text(0.50, 0.64, 'Confidence: 84.50%', color='#34d399', fontsize=10, ha='center')

# Card 3: Risk Score
c3 = patches.FancyBboxPatch(
    (0.68, 0.58),
    0.26,
    0.28,
    boxstyle='round,pad=0.02',
    edgecolor='#ef4444',
    facecolor='#1e293b',
    linewidth=1.5,
)
ax.add_patch(c3)
plt.text(
    0.81,
    0.80,
    'SYSTEMIC RISK SCORE',
    color='#94a3b8',
    fontsize=9,
    fontweight='bold',
    ha='center',
)
plt.text(
    0.81, 0.72, '94.67%', color='#ef4444', fontsize=16, fontweight='bold', ha='center'
)
plt.text(
    0.81,
    0.64,
    'High Risk Vulnerability',
    color='#fca5a5',
    fontsize=9.5,
    fontweight='bold',
    ha='center',
)

# Norwood Reference Scale Bar
guide = patches.FancyBboxPatch(
    (0.06, 0.12),
    0.88,
    0.38,
    boxstyle='round,pad=0.02',
    edgecolor='#334155',
    facecolor='#0f172a',
)
ax.add_patch(guide)
plt.text(
    0.09,
    0.44,
    'Norwood-Hamilton Staging Progression (Active State Highlighted):',
    color='#f1f5f9',
    fontsize=10.5,
    fontweight='bold',
)

types = [
    'Type 1\nJuvenile',
    'Type 2\nMild',
    'Type 3\n★ Active ★',
    'Type 4\nCrown',
    'Type 5\nAdvanced',
    'Type 6\nMerged',
    'Type 7\nSevere',
]
for idx, t in enumerate(types):
  x_pos = 0.08 + idx * 0.12
  is_active = idx == 2
  item = patches.FancyBboxPatch(
      (x_pos, 0.18),
      0.105,
      0.22,
      boxstyle='round,pad=0.01',
      edgecolor='#2dd4bf' if is_active else '#334155',
      facecolor='#134e4a' if is_active else '#1e293b',
      linewidth=2 if is_active else 1,
  )
  ax.add_patch(item)
  plt.text(
      x_pos + 0.052,
      0.28,
      t,
      color='#ffffff' if is_active else '#94a3b8',
      fontsize=8.5,
      fontweight='bold' if is_active else 'normal',
      ha='center',
  )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('fig_5_2_dashboard_analysis.png')
plt.close()
print('✓ Saved: fig_5_2_dashboard_analysis.png')

# ==========================================================
# FIG 5.3: Dashboard Personalized Care Recommendations
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_facecolor('#090d16')
fig.patch.set_facecolor('#090d16')

plt.text(
    0.5,
    0.93,
    'Personalized Care Recommendations & Vector RAG Output',
    color='#38bdf8',
    fontsize=14,
    fontweight='bold',
    ha='center',
)

card = patches.FancyBboxPatch(
    (0.06, 0.08),
    0.88,
    0.80,
    boxstyle='round,pad=0.02',
    edgecolor='#334155',
    facecolor='#1e293b',
)
ax.add_patch(card)

plt.text(
    0.10,
    0.82,
    'Personalized Clinical Recovery Protocol (Synthesized via Vector RAG):',
    color='#38bdf8',
    fontsize=11,
    fontweight='bold',
)

rec_text = """### 1. Diagnostic Summary:
• Confirmed Scalp Pathology: Alopecia Areata (Autoimmune follicular shedding)
• Staging: Norwood Type 3 (Temporal recession with active miniaturization)
• Systemic Risk Probability: 94.67% (Driven by genetics, high stress, and zinc deficiency)

### 2. Clinical Care Schedule:
• Wash Regimen: 3x weekly with ketoconazole 2% or zinc pyrithione shampoo.
• Topical Therapy: Anti-inflammatory peptide serum; consult dermatologist for corticosteroid evaluation.
• Mechanical Care: Avoid high-tension hairstyles (traction), aggressive combing, and heat styling.

### 3. Nutritional & Lifestyle Interventions:
• Dietary Intake: Supplement with elemental Zinc (15-30mg), Vitamin D3 (2000 IU), and bioavailable Biotin.
• Stress Reduction: Cortisol-lowering protocols (7-8 hours sleep, mindfulness routines).

*Medical Disclaimer: AI screening system output. Verify clinical regimen with a certified dermatologist."""

plt.text(
    0.10,
    0.14,
    rec_text,
    color="#cbd5e1",
    fontsize=9.5,
    linespacing=1.55,
    family="monospace",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('fig_5_3_dashboard_recommendations.png')
plt.close()
print('✓ Saved: fig_5_3_dashboard_recommendations.png')