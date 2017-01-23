FISH_DATA_DIR=$HOME/storage/PAN/cnn/fish2.0/
TRAIN_DIR=$HOME/storage/PAN/cnn/fish2.0/save/
EVAL_DIR=$HOME/storage/PAN/cnn/fish2.0/eval/
cd inception 
bazel-bin/inception/fish_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${FISH_DATA_DIR}" \
  --subset=validation \
  --num_examples=200 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --subset=train \
  --run_once 2>&1|grep WARNING -v
