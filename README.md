# Inception_V3
A training, validation, and predict demo by using inception v3.
### Usage

1. Install Bazel 4.0 or higer version.

2. Install Tensorflow 0.12
```sh
$ source init.sh
```

3. Build training data.

Create a tar.gz file with structure below:
  ```sh
  $ less sample.tar.gz
       rw-r--r-- worker/worker 823 2017-01-21 fish2.0/A1/953275.jpg
       rw-r--r-- worker/worker 823 2017-01-21  fish2.0/A2/1067563.jpg
       rw-r--r-- worker/worker 823 2017-01-21  fish2.0/A3/2353330.jpg
       rw-r--r-- worker/worker 823 2017-01-21  fish2.0/A4/1664412.jpg
       rw-r--r-- worker/worker 823 2017-01-21  fish2.0/A5/1264412.jpg
 ```
```sh
$  sh build_data.sh
```
4. Train the data
To make sure that you have pre-trained Inception V3 model like model.ckpt-157585.
```sh
$ sh train.sh
```
5. Now you will get Checkpoint file
```sh
-rw-r--r-- 1 worker w  72K 01-23 12:51 model.ckpt-2000.index
-rw-r--r-- 1 worker w  519 01-23 12:51 checkpoint
-rw-r--r-- 1 worker w  13M 01-23 12:52 model.ckpt-2000.meta
```
6. You can run eval.sh to see the validation result.
```sh 
$ sh eval.sh
2017-01-23 18:11:13.276846: starting evaluation on (train).
2017-01-23 18:11:35.051256: precision @ 1 = 1.0000 recall @ 5 = 1.0000 [224 examples]
```
7. Prepare to predict new data via Inception V3
```sh 
$ sh build_test_data.sh
```
You will see 
```sh 
$ ls
test-00000-of-00001
```
8. Get the prediction.
```sh 
$ sh predict.sh 10
2017-01-23 18:16:54.551870: starting evaluation on (test).
1167803.jpg 2
1343160.jpg 1
2369878.jpg 5
1343563.jpg 1
1210248.jpg 5
1220936.jpg 5
2241516.jpg 5
1440936.jpg 1
1265147.jpg 1
2305191.jpg 1
```

### Modification
1. The multi-thread problem when predict.
2. Print the file name and corresponding prediction result.
