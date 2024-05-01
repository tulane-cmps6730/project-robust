data_dir=$DATA_DIR
dataset=$1


for seed in 2
do

# 1src
for src in 0 1 2 3 
do
for trg in 0 1 2 3
do

if [ $src -ne $trg ]
then

     python -m domainbed.scripts.prompt_tta \
          --data_dir $data_dir --steps 51 \
          --dataset $dataset\
          --train_envs $src --test_envs $trg\
          --output_dir results_tta/erm_1src/$dataset_$seed/$src/tta_$trg \
          --hparams '{"prompt_dim": 4, "lr_prompt": 1e-1, "batch_size": 128}' \
          --restore results_erm/erm_1src/$dataset_$seed/$src/best_model.pkl

fi
done
done


# 3src
for trg in 0 1 2 3
do
     python -m domainbed.scripts.prompt_tta \
          --data_dir $data_dir --steps 51 \
          --dataset $dataset\
          --test_envs $trg\
          --output_dir results_tta/erm_3src/$dataset_$seed/tta_$trg \
          --hparams '{"prompt_dim": 4, "lr_prompt": 1e-1, "batch_size": 128}' \
          --restore results_erm/erm_3src/$dataset_$seed/$trg/best_model.pkl
done
done