OUTPUT_DIRECTORY=$HOME/storage/PAN/cnn/fish
TEST_DIR=$HOME/storage/PAN/cnn/fish/test_dir
LABELS_FILE=$HOME/storage/PAN/cnn/fish/labels_for_predict.txt

cd inception
bazel-bin/inception/build_image_test_data \
  --test_directory="${TEST_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=1 \
  --validation_shards=1 \
  --num_threads=1
