import matplotlib.pyplot as plt
from PIL import Image

def display_top_similar_images(query_image_path, similar_images):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    
    query_image = Image.open(query_image_path)
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for i, (img_path, similarity) in enumerate(similar_images):
        img = Image.open(img_path)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Sim: {similarity:.2f}")
        axes[i + 1].axis("off")

    plt.show()
