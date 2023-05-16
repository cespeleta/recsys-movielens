: '
Script to overfit 1 batch: useful for quickly debugging or trying to overfit on purpose. 

This is a sanity check to check that our implementation works. This will train
over and over and validate on the same batch. The network should be able to 
overfit this batch, if it does not it means that there is a bug in our implementation.
'

EXPERIMENT_NAME="overfit_batches"
TARGET="rating"

echo "Running experiment: $EXPERIMENT_NAME"
echo "Target used: $TARGET"

python src/main_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model models.mf.MatrixFactorization \
    --model.pytorch_model.n_factors 256 \
    --model.learning_rate 0.01 \
    --data.target ${TARGET} \
    --data.batch_size 12 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu \
    --trainer.overfit_batches 1 \
    --trainer.max_epochs 100
