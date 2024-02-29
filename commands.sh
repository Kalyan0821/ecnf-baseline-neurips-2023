# LL = -8.27 (510407 params)
CUDA_VISIBLE_DEVICES=0 python examples/dw4.py +model_name=egnn


# LL = -8.26 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/dw4.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# LL = -8.31 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/dw4.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# -----------------------------------------------------------------------------------

# LL = -30.5 (510407 params)
CUDA_VISIBLE_DEVICES=0 python examples/lj13.py +model_name=egnn


# LL = -30.5 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/lj13.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# LL = -30.3 (3625 params)
CUDA_VISIBLE_DEVICES=1 python examples/lj13.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# -----------------------------------------------------------------------------------

# LL = +53.4 (92487 params) [plot]
CUDA_VISIBLE_DEVICES=0 python examples/aldp.py +model_name=egnn \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# LL = +32.0 (92487 params)
CUDA_VISIBLE_DEVICES=0 python examples/aldp.py +model_name=egnn \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3


# LL = +75.7 (13537 params) --> train + eval: very slow 
CUDA_VISIBLE_DEVICES=1 python examples/aldp.py +model_name=mace \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=1e-2
# LL = +57.0 (13537 params) --> 3x slower than egnn
CUDA_VISIBLE_DEVICES=1 python examples/aldp.py +model_name=mace \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# LL = +62.3 (6497 params) --> 2x slower [plot]
CUDA_VISIBLE_DEVICES=1 python examples/aldp.py +model_name=mace \
                              flow.network.mace.correlation=2 \
                              training.use_fixed_step_size=true \
                              training.optimizer.use_schedule=false \
                              training.optimizer.init_lr=5e-3
# -----------------------------------------------------------------------------------