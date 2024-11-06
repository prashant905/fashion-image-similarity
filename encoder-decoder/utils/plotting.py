import matplotlib.pyplot as plt
from PIL import Image

def plot_similarity_results(results, figsize=(15, 4)):
    n_images = len(results)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    fig.suptitle('Similar Images (with Distance Scores)', fontsize=14, y=1.05)

    if n_images == 1:
        axes = [axes]

    for idx, (ax, result) in enumerate(zip(axes, results)):
        img = Image.open(result['image_path'])
        ax.imshow(img)

        distance = result['distance']
        distance_text = f'Distance: {distance:.2f}'
        ax.set_title(f'#{idx + 1}\n{distance_text}', pad=10)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig
