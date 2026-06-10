import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# 掩码区域
ax.add_patch(plt.Rectangle((30, 30), 40, 40, linewidth=2, edgecolor='b', facecolor='lightblue', alpha=0.5))

# 标注
ax.text(50, 75, "Mask region", fontsize=12, color="b", ha="center")

plt.title("Top-down Projection of 3D Mask")
plt.show()
