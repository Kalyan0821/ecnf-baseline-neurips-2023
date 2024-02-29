# NLL = 8.27 (510407 params)
CUDA_VISIBLE_DEVICES=0 python examples/dw4.py +model_name=egnn
# NLL = 8.31 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/dw4.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# NLL = 8.26
CUDA_VISIBLE_DEVICES=1 python examples/dw4.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# -----------------------------------------------------------------------------------


# NLL = 30.5 (510407 params)
CUDA_VISIBLE_DEVICES=0 python examples/lj13.py +model_name=egnn
# NLL = 30.3 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/lj13.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# NLL = 30.5
CUDA_VISIBLE_DEVICES=1 python examples/lj13.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# -----------------------------------------------------------------------------------


# NLL = -20.1 (92487 params)
CUDA_VISIBLE_DEVICES=0 python examples/aldp.py +model_name=egnn
# NLL = -31.7 / -20.2 ??
CUDA_VISIBLE_DEVICES=0 python examples/aldp.py +model_name=egnn \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# NLL = -39.4 / -20.3 ??
CUDA_VISIBLE_DEVICES=0 python examples/aldp.py +model_name=egnn \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2


# NLL = -51.3 / -20.1 ??  (13537 params)  --> train + eval: 4x slower than egnn
CUDA_VISIBLE_DEVICES=1 python examples/aldp.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# NLL = -68.03 / -20.1 ??
CUDA_VISIBLE_DEVICES=1 python examples/aldp.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# -----------------------------------------------------------------------------------