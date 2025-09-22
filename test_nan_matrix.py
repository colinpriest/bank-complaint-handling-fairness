#!/usr/bin/env python3
"""Test the NaN effectiveness matrix implementation"""

import numpy as np
import matplotlib.pyplot as plt

# Test the NaN matrix implementation
strategies = ['strategy1', 'strategy2', 'strategy3']
models = ['model1', 'model2', 'model3']

# Create matrix with NaN values
effectiveness_matrix = np.full((len(strategies), len(models)), np.nan)

# Add some test data to show mixed NaN and real values
effectiveness_matrix[0, 0] = 75.5  # One real value
effectiveness_matrix[1, 2] = 82.3  # Another real value

print("Effectiveness Matrix:")
print(effectiveness_matrix)
print(f"\nAll NaN?: {np.all(np.isnan(effectiveness_matrix))}")
print(f"Any NaN?: {np.any(np.isnan(effectiveness_matrix))}")

# Test the visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Create masked array for visualization
masked_array = np.ma.array(effectiveness_matrix, mask=np.isnan(effectiveness_matrix))

# Plot with colormap
im = ax.imshow(masked_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(models)))
ax.set_yticks(range(len(strategies)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_yticklabels(strategies)

# Add text annotations
for i in range(len(strategies)):
    for j in range(len(models)):
        if np.isnan(effectiveness_matrix[i, j]):
            ax.text(j, i, 'N/A',
                   ha='center', va='center', fontweight='bold',
                   color='gray', fontsize=10)
        else:
            ax.text(j, i, f'{effectiveness_matrix[i, j]:.1f}%',
                   ha='center', va='center', fontweight='bold')

# Add colorbar and labels
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Effectiveness (%)', fontweight='bold')

ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('Strategy', fontweight='bold')

# Title based on data availability
if np.all(np.isnan(effectiveness_matrix)):
    ax.set_title('Effectiveness Matrix\n(No data available)',
                fontweight='bold', color='darkred')
else:
    ax.set_title('Effectiveness Matrix\n(Partial data)',
                fontweight='bold')

plt.tight_layout()
plt.savefig('test_nan_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTest completed successfully!")
print("Check test_nan_matrix.png for visualization")