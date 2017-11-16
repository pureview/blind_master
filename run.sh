# Calculate accuracy of different configuration

# var mistake
for m in 0.6 0.7 0.8 0.9
do
    python mnist.py -drop_prob 1 -mistake $m 
    python mnist.py -drop_prob 0.5 -mistake $m
    for dim in 10 50 100 200 500 1000
    do
        python mnist.py -drop_prob 1 -mistake $m -vector -out_dim $dim
        python mnist.py -drop_prob 1 -mistake $m -vector -out_dim $dim -dynamic_lr
    done
done
