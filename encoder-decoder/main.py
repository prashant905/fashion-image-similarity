from models.similarity_search import ImageSimilaritySearch
from utils.plotting import plot_similarity_results

def main():
    searcher = ImageSimilaritySearch(latent_dim=128)
    searcher.train(image_dir="fashion-dataset/sample/", epochs=50)
    searcher.build_index("fashion-dataset/sample")
    
    results = searcher.search("fashion-dataset/images/36795.jpg", k=5)
    plot_similarity_results(results)

if __name__ == "__main__":
    main()
