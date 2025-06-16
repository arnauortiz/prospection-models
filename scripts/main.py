import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_loading import main_dataset
from visualization.statistics import statistics, tsne_visualization
from modelling.unsupervised import main_unsupervised
from modelling.supervised import main_supervised

if __name__ == "__main__":
    dataset = main_dataset()
    statistics(dataset, "output/statistics.png")
    tsne_visualization(dataset, "output/tsne_visualization.png")
    main_unsupervised(dataset)
    main_supervised(dataset)
    
    
