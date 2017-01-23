OUTPUT_DIRECTORY=$HOME/storage/PAN/cnn/
TRAIN_DIR=$HOME/storage/PAN/cnn/train_dir/
VALIDATION_DIR=$HOME/storage/PAN/cnn/valid_dir/
LABELS_FILE=$HOME/storage/PAN/cnn/labels.txt
INCEPTION_MODEL_DIR=$HOME/storage/PAN/cnn/fish/model/inception-v3
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"
FISH_DATA_DIR=$HOME/storage/PAN/cnn/fish
TRAIN_DIR=$HOME/storage/PAN/cnn/fish/save

cd ~/storage/PAN/tensorflow/
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=$FISH_DATA_DIR/bottlenecks \
--how_many_training_steps 500 \
--model_dir=$FISH_DATA_DIR/model \
--output_graph=$FISH_DATA_DIR/moel/retrained_graph.pb \
--output_labels=$FISH_DATA_DIR/retrained_labels.txt \
--image_dir /tf_files/flower_photos
