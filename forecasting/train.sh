
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
PARAM_COMBO=11
CHECKPOINTDIR=../model_checkpoints/
CHECKPOINT=1
NUM_EPOCHS=100
OUTDIR=../results/
python3 ./model.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
