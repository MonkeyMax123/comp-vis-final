from sklearn.model_selection import train_test_split
from Build_Histogram import *
from Detect_Feature_And_KeyPoints import *
from Load_Dataset_folder import *
from Features_Processing import *
from Linear_Processsing_Pipeline import *
from Training_Poly_Processing_Pipeline import *
from Testing_Poly_Processing_Pipeline import *


root_path = "C:\\Users\\Max\\dev\\comp vis final\\"
data_path = root_path + 'Modern Shark Teeth'


image_files, labels = load_dataset_folder(data_path)
features, processed_labels = Features_Processing(image_files, labels)


features_train, features_test, labels_train, labels_test = train_test_split(features, processed_labels, test_size = .2, random_state = 0)
features_train_train, features_validation, labels_train_train, labels_validation = train_test_split(features_train, labels_train, test_size = .25, random_state = 0)


c, d = Poly_Processing_Pipeline(features_train_train, features_validation, labels_train_train, labels_validation)


training_accuracy = Testing_Poly_SVC(features_train_train, features_validation, labels_train_train, labels_validation, c, d)
testing_accuracy = Testing_Poly_SVC(features_train, features_test, labels_train, labels_test,c, d)

print("c: ", c)
print("d: ", d)
print("Training Accuracy: ", training_accuracy)
print("Testing Accuracy: ", testing_accuracy)