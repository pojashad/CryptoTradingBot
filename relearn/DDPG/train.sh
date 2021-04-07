###Install requirements first:
#pip3 install requirements.txt

TRAIN_DATA=../trade_df.csv
OUTDIR=./
python3 ./model.py --train_data $TRAIN_DATA --outdir $OUTDIR
