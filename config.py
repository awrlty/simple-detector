DATA_PATH = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner"
TEST_PATH = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\test"
OUTPUT_PATH = 'results/corner_detector'

NUM_WORKERS = 0
BATCH_SIZE = 8
EPOCHS = 1500
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON = 1E-6
IMAGE_SIZE = 1024  # square only (no rectangle)

S = 32       # Divide each image into a SxS grid (must fix)
B = 2       # Number of bounding boxes to predict
C = 1      # Number of classes in the dataset
