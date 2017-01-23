OUTPUT_DIRECTORY=$HOME/storage/PAN/cnn
TEST_DIR=$HOME/storage/PAN/cnn/fish
FISH_DATA_DIR=$HOME/storage/PAN/cnn/fish/
TRAIN_DIR=$HOME/storage/PAN/cnn/fish2.0/save/
EVAL_LOG_DIR=$HOME/storage/PAN/cnn/fish/eval/
cd inception 
bazel-bin/inception/fish_predict \
  --eval_dir="${EVAL_LOG_DIR}" \
  --data_dir="${FISH_DATA_DIR}" \
  --num_examples=200 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --subset=test \
  --batch_size=$1 \
  --run_once 2>&1|grep WARNING -v|grep "Please" -v|grep "Instructions" -v
