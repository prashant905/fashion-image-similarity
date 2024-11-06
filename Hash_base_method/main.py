from image_search import ImageSimilaritySearch

if __name__ == "__main__":
    DATASET_PATH = 'data/sample/'
    QUERY_IMAGE = 'data/images/15970.jpg'

    image_search = ImageSimilaritySearch(
        image_folder=DATASET_PATH,
        hash_dim=128,
        num_permutations=128
    )
    image_search.index_images()
    image_search.find_similar_images(
        QUERY_IMAGE,
        num_results=5,
        visualize=True
    )
