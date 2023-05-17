for expected_batchsize in 1000
do

for epsilon in 4
do

for EPOCH in 25
do

for lr in 1
do

python main.py --expected_batchsize $expected_batchsize --epsilon $epsilon --EPOCH $EPOCH --lr $lr --log_dir logs 


done
done
done
done


# srun --time=08:00:00 --reservation=A100 --gres=gpu:a100:1  --mem=50G --resv-ports=1 --pty /bin/bash -l
# jupyter notebook --no-browser --ip=0.0.0.0
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9