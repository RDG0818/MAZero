#!/bin/bash

env='smac'
scenario='3m'
exp_name="3m"

K=1
N=10

seeds=(0)
mcts_rhos=(0.75)
awac_lambdas=(3)
adv_clips=(3.0)

for seed in "${seeds[@]}"; do
  for mcts_rho in "${mcts_rhos[@]}"; do
    for awac_lambda in "${awac_lambdas[@]}"; do
      for adv_clip in "${adv_clips[@]}"; do

        run_name="${exp_name}_seed${seed}_rho${mcts_rho}_lambda${awac_lambda}_clip${adv_clip}"

        echo "Running: $run_name"

        python main.py --opr train --case $env --env_name $scenario --exp_name $run_name --seed $seed \
          --num_cpu 24 --num_gpus 1 --train_on_gpu --reanalyze_on_gpu --selfplay_on_gpu \
          --data_actors 1 --num_pmcts 4 --reanalyze_actors 4 \
          --test_interval 500 --test_episodes 32 --target_model_interval 200 \
          --batch_size 256 --num_simulations $N --sampled_action_times $K \
          --training_steps 49000 --last_step 1000 --lr 1e-4 --lr_adjust_func const --max_grad_norm 5 \
          --total_transitions 2000000 --start_transition 500 --discount 0.99 \
          --target_value_type pred-re --revisit_policy_search_rate 1.0 --use_off_correction \
          --value_transform_type vector --use_mcts_test \
          --use_priority --use_max_priority \
          --PG_type sharp --awac_lambda $awac_lambda --adv_clip $adv_clip \
          --mcts_rho $mcts_rho --mcts_lambda 0.8

      done
    done
  done
done