import numpy as np


class KnnModel:
    def __init__(self):
        return

    def __init__(self, training_set, training_labels, count_neighbor):
        self.trainingSet = training_set
        self.trainingLabels = training_labels
        self.count_neighbor = count_neighbor
        return

    # def evaluate(self, feature_matrix_test, labels_test, param):
    #     return

    def predict(self, feature_vector_input):
        max_label = self.trainingLabels[0]
        num_samples = self.trainingSet.shape[0]
        repeat_input = np.tile(feature_vector_input, (num_samples, 1))
        diff = repeat_input - self.trainingSet
        squared_diff = diff ** 2
        squared_dist = np.sum(squared_diff, axis=1)
        distance = squared_dist ** 0.5
        sorted_dist_indices = np.argsort(distance)
        class_count = {}
        for i in range(self.count_neighbor):
            cur_index = sorted_dist_indices[i]
            cur_label = self.trainingLabels[cur_index]
            class_count[cur_label] = class_count.get(cur_label, 0) + 1
        max_count = 0
        for key, value in class_count.items():
            if value > max_count:
                max_count = value
                max_label = key
        return max_label

    # def batch_predict(self, feature_vector_inputs, label_inputs, param):
    #     return

