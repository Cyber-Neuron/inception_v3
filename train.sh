LABELS_FILE=$HOME/storage/PAN/cnn/fish2.0/raw-data/labels.txt
INCEPTION_MODEL_DIR=$HOME/storage/PAN/cnn/fish/model/inception-v3
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"
FISH_DATA_DIR=$HOME/storage/PAN/cnn/fish2.0/
TRAIN_DIR=$HOME/storage/PAN/cnn/fish2.0/save/
cd inception 
bazel-bin/inception/fish_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FISH_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
