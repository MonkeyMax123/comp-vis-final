from sklearn.model_selection import train_test_split
from build_histogram import *
from Detect_Feature_And_KeyPoints import *
from Load_Dataset_folder import *
from Features_Processing import *
from Processsing_Pipeline import *


root_path = "C:\\Users\\Max\\dev\\comp vis final\\"
data_path = root_path + 'Modern Shark Teeth'


image_files, labels = load_dataset_folder(data_path)
features, processed_labels = Features_Processing(image_files, labels)


features_train, features_test, labels_train, labels_test = train_test_split(features, processed_labels, test_size = .2, random_state = 0)
features_train_train, features_validation, labels_train_train, labels_validation = train_test_split(features_train, labels_train, test_size = .25, random_state = 0)


validation_accuracy = Processing_Pipeline(features_train_train, features_validation, labels_train_train, labels_validation)
testing_accuracy = Processing_Pipeline(features_train, features_test, labels_train, labels_test)

print("Validation Accuracy: ", validation_accuracy)
print("Testing Accuracy: ", testing_accuracy)