import pathlib
import os

working_directory = os.path.dirname(os.path.realpath(__file__))
data_path = pathlib.Path(os.path.join(working_directory, "data/"))

# Preprocessing
distance_matrix_dir = pathlib.Path(os.path.join(data_path, "distance_matrix/"))

raw_data_path = pathlib.Path(os.path.join(data_path, "raw_data.csv"))
clean_and_filtered_data_path = pathlib.Path(os.path.join(data_path, "clean_and_filtered_data.csv"))
rides_data_path = pathlib.Path(os.path.join(data_path, "rides_data.csv"))
stops_definition_path = pathlib.Path(os.path.join(data_path, "stops.json"))

grouped_data_path = pathlib.Path(os.path.join(data_path, "grouped_data.csv"))

# Training
train_data_path = pathlib.Path(os.path.join(data_path, "train_data.csv"))
validation_data_path = pathlib.Path(os.path.join(data_path, "validation_data.csv"))

models_dir = pathlib.Path(os.path.join(data_path, "models/"))
