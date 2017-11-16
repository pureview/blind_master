# Calculate accuracy of different configuration

# var mistake
for m in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 
do
    for dim in 10 50 100 200 500 1000 
    do
        echo $m $dim
    done
done
