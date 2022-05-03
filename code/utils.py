from collections import Counter
import numpy as np


def calculate_class_weights(train_generator):
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weight = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    return class_weight