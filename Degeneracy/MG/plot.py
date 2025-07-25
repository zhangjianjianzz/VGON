import os
import sys
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Configuration
# ------------------------
DATA_NAME = 'Degene_MG'
RESULTS_DIR = os.path.join(sys.path[0], 'results')
MAT_PATH = os.path.join(RESULTS_DIR, f'{DATA_NAME}.mat')
SAVE_PATH = os.path.join(RESULTS_DIR, f'Rain_MG.png')
COLORS = ["#6FAE45", "#589CD6", "#4474C4", "#660874", "#AB2A3C"]

# ------------------------
# Load and prepare data
# ------------------------
data = sio.loadmat(MAT_PATH)
df = pd.DataFrame({
    r'$|v_1 \rangle$': data['m'][:, 0],
    r'$|v_2 \rangle$': data['m'][:, 1],
    r'$|v_3 \rangle$': data['m'][:, 2],
    r'$|v_4 \rangle$': data['m'][:, 3],
    r'$|v_5 \rangle$': data['m'][:, 4],
})
print('Overlap statistics:\n', df.describe())

# ------------------------
# Plotting
# ------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('(a)', x=0.1)

# Raincloud plot
pt.half_violinplot(data=df, inner=None, width=0.6, palette=COLORS, edgecolor="white", ax=ax)
sns.stripplot(data=df, jitter=True, edgecolor="none", size=1.2, palette=COLORS, ax=ax)
sns.boxplot(data=df, width=0.15, zorder=10, whiskerprops={'linewidth': 1.5, "zorder": 10},
            boxprops={'facecolor': 'none', "zorder": 10}, palette=COLORS, ax=ax)

# Axis settings
ax.set_ylabel('Overlap')
ax.set_ylim(-0.1, 1)
ax.grid(True)

# ------------------------
# Save figure
# ------------------------
plt.savefig(SAVE_PATH, format='png', bbox_inches='tight')
