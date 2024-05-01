# PACS
# ['art_painting', 'cartoon', 'photo', 'sketch']
# 1src
data_dir=$DATA_DIR
dataset=$1

for seed in 0
do
# 1src
for src in 0 1 2 3 
do
    python -m domainbed.scripts.train\
        --data_dir $data_dir --steps 3001\
        --seed $seed\
        --dataset $dataset --train_envs $src\
        --algorithm ERM --output_dir results_erm/erm_1src/pacs_$seed/${pacs[$src]}\
        --hparams '{"lr": 5e-6, "lr_classifier": 5e-5}'
done




# 3src
for trg in 0 1 2 3
do
    python -m domainbed.scripts.train\
        --data_dir $data_dir --steps 3001\
        --dataset $dataset --test_env $trg\
        --seed $seed\
        --algorithm ERM --output_dir results_erm/erm_3src/pacs_$seed/${pacs[$trg]}\
        --hparams '{"lr": 5e-6, "lr_classifier": 5e-5}'
done

done
