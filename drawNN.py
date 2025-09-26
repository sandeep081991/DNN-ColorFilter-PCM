import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_labels):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # Draw neurons
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            if layer_size > 10 and n == len(layer_sizes) - 1:
                # Collapse output layer (e.g., 150 neurons to 5)
                if m in [0, 1, 2, 3, 4]:  # Draw only 5 sample nodes
                    circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 0.2, color='skyblue', ec='k', zorder=4)
                    ax.add_artist(circle)
                    if layer_labels[n][m] != '':
                        ax.text(n*h_spacing + left + 0.25, layer_top - m*v_spacing,
                                layer_labels[n][m], fontsize=8, va='center')
                continue
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 0.2, color='skyblue', ec='k', zorder=4)
            ax.add_artist(circle)
            if layer_labels[n][m] != '':
                ax.text(n*h_spacing + left - 0.5, layer_top - m*v_spacing,
                        layer_labels[n][m], fontsize=8, va='center')

    # Draw lines
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(min(layer_size_b, 5) if layer_size_b > 10 else layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n+1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                  c='gray')
                ax.add_artist(line)

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Network layers
layer_sizes = [4, 64, 128, 5]  # Collapsed 150 to 5 for drawing
layer_labels = [
    ['SiO2', 'Sb2S3', 'Phase', 'Angle'],
    ['']*64,
    ['']*128,
    ['T(λ₁)', 'T(λ₅₀)', '...', 'T(λ₁₀₀)', 'T(λ₁₅₀)']  # Sample outputs
]

draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, layer_labels)

# Layer titles
layer_titles = ['Input Layer', 'Hidden Layer 1 (ReLU)', 'Hidden Layer 2 (ReLU)', 'Output Layer (Linear)']
for i, title in enumerate(layer_titles):
    ax.text(i * 0.27 + 0.1, 0.95, title, fontsize=10, ha='center', weight='bold')

plt.title('Neural Network Architecture for Transmission Prediction', fontsize=13, weight='bold')
plt.show()
