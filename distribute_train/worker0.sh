OUTPUT_DIRECTORY=$HOME/storage/PAN/cnn/
TRAIN_DIR=$HOME/storage/PAN/cnn/train_dir/
VALIDATION_DIR=$HOME/storage/PAN/cnn/valid_dir/
LABELS_FILE=$HOME/storage/PAN/cnn/labels.txt
INCEPTION_MODEL_DIR=$HOME/storage/PAN/cnn/fish/model/inception-v3
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"
FISH_DATA_DIR=$HOME/storage/PAN/cnn/fish/
TRAIN_DIR=$HOME/storage/PAN/cnn/fish/save/

cd inception
bazel-bin/inception/fish_distributed_train \
--train_dir="${TRAIN_DIR}" \
--data_dir="${FISH_DATA_DIR}" \
--pretrained_model_checkpoint_path="${MODEL_PATH}" \
--fine_tune=True \
--initial_learning_rate=0.001 \
--input_queue_memory_factor=1 \
--batch_size=32 \
--job_name='worker' \
--task_id=0 \
--ps_hosts='lg-1r14-n04:8899' \
--worker_hosts='aw-4r14-n30:8899,sw-2r02-n28:8899'
