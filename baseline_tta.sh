dataset=$1

for seed in 1 2
do

for src in 0 1 2 3 
do
for trg in 0 1 2 3
do

if [ $src -ne $trg ]
then
     # 1src
     python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_1src/pacs_$seed/${pacs[$src]}\
        --test_envs $trg\
        --output_dir results_tent/erm_1src/pacs_$seed/${pacs[$src]}/tta_${pacs[$trg]} \
        --adapt_algorithm T3A 

    python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_1src/pacs_$seed/${pacs[$src]}\
        --test_envs $trg\
        --output_dir results_tent/erm_1src/pacs_$seed/${pacs[$src]}/tta_${pacs[$trg]} \
        --adapt_algorithm TentPreBN

    python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_1src/pacs_$seed/${pacs[$src]}\
        --test_envs $trg\
        --output_dir results_tent/erm_1src/pacs_$seed/${pacs[$src]}/tta_${pacs[$trg]} \
        --adapt_algorithm TentClf
          

fi
done
done


for trg in 0 1 2 3
do

     python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_3src/pacs_$seed/${pacs[$trg]}\
        --test_envs $trg\
        --output_dir results_tent/erm_3src/pacs_$seed/tta_${pacs[$trg]} \
        --adapt_algorithm T3A 

    python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_3src/pacs_$seed/${pacs[$trg]}\
        --test_envs $trg\
        --output_dir results_tent/erm_3src/pacs_$seed/tta_${pacs[$trg]} \
        --adapt_algorithm TentPreBN

    python -m domainbed.scripts.tta \
        --input_dir results_erm/erm_3src/pacs_$seed/${pacs[$trg]}\
        --test_envs $trg\
        --output_dir results_tent/erm_3src/pacs_$seed/tta_${pacs[$trg]} \
        --adapt_algorithm TentClf

done

done