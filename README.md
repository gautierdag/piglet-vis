# hog
piglet with eyes

To reproduce Piglet:

`python main.py --config-name piglet`


To run our image baseline:

`python main.py --config-name image`


`run_experiment -b train_job.sh -e experiments.txt \ --cpus-per-task=6 --gres=gpu:1 --mem-per-cpu=8G --time=1-00:00:00`