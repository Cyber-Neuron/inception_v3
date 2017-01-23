cd inception 
bazel-bin/inception/fish_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts='lg-1r14-n04:8899' \
--worker_hosts='aw-4r14-n30:8899,sw-2r02-n28:8899'
