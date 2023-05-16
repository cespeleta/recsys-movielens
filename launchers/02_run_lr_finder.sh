: '
Find optimal learning rate using LearningRateFinder for the following models:

1. Run MatrixFactorizationRaw with raw output
2. Run MatrixFactorization with sigmoid in the output
3. Run MatrixFactorizationWithBias

After the first run, copy the optimal learning rate and update the script call.

If we want to log our experiments properly run again all the models with the correct
learning rate, otherwise pytorch lightning will store in the logs the default values.

Instructions:
- Go to the config file and make sure that the LearningRateFinder is active
- Execute the script
> ./launchers/02_run_lr_finder.sh

Docs: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
'
# Define variables
EXPERIMENT_NAME="lr_finder"
TARGET="rating"

echo "Running experiment: $EXPERIMENT_NAME"
echo "Target used: $TARGET"

python src/main_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model models.mf.MatrixFactorization \
    --model.learning_rate 0.01278 \
    --data.target ${TARGET} \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.callbacks+=LearningRateFinder \
    --trainer.callbacks.min_lr=0.001 \
    --trainer.callbacks.max_lr=0.5

python src/main_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model models.mf_with_bias.MatrixFactorizationWithBias \
    --model.learning_rate 0.0120 \
    --data.target ${TARGET} \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu \
    --trainer.callbacks+=LearningRateFinder \
    --trainer.callbacks.min_lr=0.001 \
    --trainer.callbacks.max_lr=0.5

python src/main_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model models.neu_mf.NeuMF \
    --model.learning_rate 0.0025 \
    --data.target ${TARGET} \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu \
    --trainer.callbacks+=LearningRateFinder \
    --trainer.callbacks.min_lr=0.001 \
    --trainer.callbacks.max_lr=0.5
