# -8.27 (510407 params)
CUDA_VISIBLE_DEVICES=0 python examples/dw4.py +model_name=egnn
# -8.26 (3625 params)
CUDA_VISIBLE_DEVICES=0 python examples/dw4.py +model_name=mace \
                              training.use_fixed_step_size=false \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2

