for runseed in 0 1 2 3 4 5 6 7 8 9
do
python split.py --ds rabbit --rs $runseed
python split.py --ds rat --rs $runseed
done
