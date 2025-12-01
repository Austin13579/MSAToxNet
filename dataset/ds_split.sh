for runseed in 0 1 2 3 4 5 6 7 8 9
do
python split.py --ds Rabbit --rs $runseed
python split.py --ds Rat --rs $runseed
done
