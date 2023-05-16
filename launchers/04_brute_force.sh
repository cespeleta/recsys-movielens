: '
Brute force optimization:
Params: 
- learning rate
- batch_size
'

EXPERIMENT_NAME="embedding_dim_earlystopping_rmse"

echo "Running experiment: $EXPERIMENT_NAME"

echo "==================="
echo "Embedding size: 64"
echo "==================="
python src/main_cli.py \
    --config configs/config_mf.yaml \
    --model.pytorch_model models.mf_with_bias.MatrixFactorizationWithBias \
    --model.pytorch_model.n_factors 64 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu

echo "==================="
echo "Embedding size: 128"
echo "==================="
python src/main_cli.py \
    --config configs/config_mf.yaml \
    --model.pytorch_model models.mf_with_bias.MatrixFactorizationWithBias \
    --model.pytorch_model.n_factors 128 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu

echo "==================="
echo "Embedding size: 256"
echo "==================="
python src/main_cli.py \
    --config configs/config_mf.yaml \
    --model.pytorch_model models.mf_with_bias.MatrixFactorizationWithBias \
    --model.pytorch_model.n_factors 256 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.accelerator cpu
