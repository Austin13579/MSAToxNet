# Efficiently Predicting Acute Dermal Toxicity with Multiscale Information and Weighted Model Averaging Strategy

This is a PyTorch implementation of MSAToxNet.

## Data splitting

Before running the codes, you need to split the datasets according to your requirements.
```
cd dataset/
python split.py --ds Rabbit/Rat --rs 0/1/2/3/4/5/6/7/8/9
```

`rs` denotes random seed.

For convenience, you can use the `ds_split.sh`
```
./ds_split.sh
```

## Training

```
cd src/
python train.py --ds Rabbit/Rat --rs 0/1/2/3/4/5/6/7/8/9
```

For convenience, you can use the `run.sh`
```
./run.sh
```
