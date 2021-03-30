

TRAIN_DATA=../trading.csv
NUM_EPOCHS=1
OUTDIR=./results/
python3 ./model.py --train_data $TRAIN_DATA --num_epochs $NUM_EPOCHS --outdir $OUTDIR