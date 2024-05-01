# [OT3P: Optimal-Transport guided Test-Time Adaptation for Vision-Language Models](https://docs.google.com/presentation/d/1Z65LXU6kdW3wt3b9PXE5-DFxuurNGz3e7i-bBIPssi4/edit?usp=sharing)

This README provides an overview and usage instructions for the shell scripts used in this project. Each script is designed to perform specific machine learning tasks using the DomainBed framework for domain adaptation.

#### - Codes

1. **erm.sh** - Trains models using Empirical Risk Minimization (ERM) on single and multiple source environments. The script accepts dataset name and data directory as inputs. To run the script, enter the command below in the terminal.

   ```bash
   DATA_DIR=<path_to_data> ./erm.sh <dataset_name>
   ```

2. **baseline_tta.sh** - Performs test-time adaptation using different algorithms on the PACS dataset after ERM model is trained. This script iterates over different seeds and environments to apply test-time adaptation (TTA) methods like T3A and Tent on models trained under the ERM framework.  To run the script, enter the command below in the terminal.

   ```bash
   bash ./baseline_tta.sh <dataset_name>
   ```

3. **OT3P.sh** - Code for our proposed method. It applies prompt-based test-time adaptation using a trained ERM model. It requires specifying a data directory and a dataset. To run the script, enter the command below in the terminal.

   ```bash
   DATA_DIR=<path_to_data> ./OT3P.sh <dataset_name>
   ```

##### Note: our codes are adapted from the DomainBed repository. You can find the original code at https://github.com/facebookresearch/DomainBed

-----

#### - Datasets

1. **Vision Task Dataset**: 

   We use PACS for our vision classification task. You can download the dataset at https://datasets.activeloop.ai/docs/ml/datasets/pacs-dataset/

2. **NLP Task Dataset**:

   We use SST5 and Yelp datasets for our NLP classification task. You can download them at the following links, respectively. 

   - Yelp: https://www.yelp.com/dataset
   - SST5: https://nlp.stanford.edu/sentiment/


