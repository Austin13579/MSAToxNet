for dataset in Rat Rabbit
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python -u train.py --rs $runseed --ds $dataset
done
done
