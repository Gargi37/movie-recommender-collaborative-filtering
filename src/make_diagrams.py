"""Architecture diagrams for the synopsis and report: Level-0 data flow diagram and system block diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

NAVY, BLUE, GREEN, AMBER, GREY = '#1F3A5F', '#4C78A8', '#5B9E6F', '#E0A030', '#6B7280'


def arrow(ax, a, b, text=None, off=(0, 0.12), color=GREY, rad=0.0, fs=8.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, color=color, lw=1.6, connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha='center', va='bottom', fontsize=fs, color='#111827')


def box(ax, x, y, w, h, text, fc, fs=9, tc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='#111827', lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, wrap=True)


# ---------------------------------------------------------------- DFD level 0
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis('off')
ax.add_patch(Circle((5.5, 3), 1.35, fc=BLUE, ec='#111827', lw=1.4))
ax.text(5.5, 3, '0\nMovie\nRecommendation\nSystem', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
box(ax, 0.2, 2.45, 1.9, 1.1, 'External entity\nViewer\n(streaming user)', GREEN, 9.5)
box(ax, 4.5, 4.9, 2.0, 0.9, 'External entity\nData source\n(provided dataset.csv)', AMBER, 9.5, tc='#111827')
box(ax, 8.9, 2.45, 1.9, 1.1, 'External entity\nMentor / Reviewer', GREEN, 9.5)
arrow(ax, (2.1, 3.25), (4.2, 3.25), 'user_id, k, method,\ndiversify option', off=(0, 0.1))
arrow(ax, (4.2, 2.75), (2.1, 2.75), 'ranked recommendations,\nexplanations, diversity score', off=(0, -0.62))
arrow(ax, (5.5, 4.9), (5.5, 4.35), 'ratings (user, movie, rating, time),\nmovie catalogue', off=(2.0, -0.28))
arrow(ax, (6.8, 3.0), (8.9, 3.0), 'evaluation report,\ndiversity / fairness audit', off=(0, 0.1))
ax.text(5.5, 0.35, 'Level-0 data flow diagram: one process, three external entities, four data flows', ha='center', fontsize=9, style='italic', color=GREY)
plt.tight_layout(); plt.savefig('reports/figures/arch_dfd_level0.png', dpi=170); plt.close()

# ---------------------------------------------------------------- block diagram
fig, ax = plt.subplots(figsize=(13, 7.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.6); ax.axis('off')
ax.text(0.15, 7.25, 'DATA PIPELINE', fontsize=10, fontweight='bold', color=NAVY)
row1 = [('Provided\ndataset.csv\n(500 rows)', AMBER, '#111827'), ('generate_data.py\ncatalogue (126 films) +\nsynthetic ratings\n= 5,000 rows', BLUE, 'white'), ('clean.py\nchecks, types,\nlog (tests prove rules)', BLUE, 'white'),
        ('make_split.py\nper-user chronological\ntrain_core / val /\ntrain_full / test (locked)', BLUE, 'white')]
for i, (t, c, tc) in enumerate(row1):
    box(ax, 0.3 + i * 3.2, 5.6, 2.5, 1.35, t, c, 9, tc)
    if i: arrow(ax, (0.3 + (i - 1) * 3.2 + 2.5, 6.28), (0.3 + i * 3.2, 6.28))
ax.text(0.15, 5.2, 'MODELLING (fit on training data only)', fontsize=10, fontweight='bold', color=NAVY)
ax.add_patch(FancyBboxPatch((0.15, 3.4), 9.5, 1.65, boxstyle='round,pad=0.02,rounding_size=0.08', fc='none', ec=GREEN, lw=1.6, ls='--'))
models = ['Baselines\nglobal / user / item mean,\nbias, popularity', 'Item-based CF\n(adjusted cosine,\nshrinkage, damping)', 'User-based CF\n(Pearson / cosine)', 'SVD matrix\nfactorisation\n(Surprise)']
for i, t in enumerate(models):
    box(ax, 0.3 + i * 2.35, 3.55, 2.1, 1.35, t, GREEN, 9)
arrow(ax, (10.0, 6.0), (9.0, 5.05), 'training data', off=(-0.2, 0.25), fs=8)
arrow(ax, (9.65, 4.22), (10.4, 4.22))
box(ax, 10.4, 3.55, 2.4, 1.35, 'Tuning: 2 rolling folds\nCF + popularity blend\nMMR diversity re-ranker', NAVY, 9)
ax.text(0.15, 3.1, 'EVALUATION AND DELIVERY', fontsize=10, fontweight='bold', color=NAVY)
box(ax, 0.3, 0.9, 3.4, 1.75, 'Evaluation and audit\n(test set opened once)\nRMSE, MAE, P/R/NDCG@K,\ncoverage, ILD, Gini,\nfairness, bootstrap CIs', AMBER, 9, '#111827')
box(ax, 4.6, 0.9, 3.4, 1.75, 'Service layer\nsrc/serving.py\nrecommend, similar,\nhistory, explain\n(cold-start fallback)', BLUE, 9)
box(ax, 9.0, 1.75, 3.8, 0.85, 'Flask REST API  (api/app.py)', GREEN, 9.5)
box(ax, 9.0, 0.6, 3.8, 0.85, 'Streamlit dashboard  (app/dashboard.py)', GREEN, 9.5)
arrow(ax, (2.0, 3.4), (2.0, 2.65), 'tuned models', off=(0.85, -0.05), fs=8)
arrow(ax, (10.9, 3.55), (7.2, 2.65), 'saved models (joblib)', off=(0.5, 0.25), fs=8)
arrow(ax, (8.0, 2.1), (9.0, 2.15)); arrow(ax, (8.0, 1.5), (9.0, 1.05))
ax.text(6.5, 0.2, 'Block diagram. Arrows show data / control flow; the service layer is shared by the API and the dashboard.', ha='center', fontsize=9, style='italic', color=GREY)
plt.tight_layout(); plt.savefig('reports/figures/arch_block_diagram.png', dpi=170); plt.close()
print('diagrams written')
