

import argparse
import collections
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import sys
import time
import uuid
import ot

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision

from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.torchmisc import dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="tta",
        choices=["domain_generalization", "domain_adaptation", "tta"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--train_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    args = parser.parse_args()
    
    if args.dataset == 'PACS' or args.dataset == 'VLCS' or args.dataset == 'OfficeHome':
        envs = [0, 1, 2, 3]
        if args.train_envs == [-1]:
            args.train_envs = [env for env in envs if env not in args.test_envs]
        if args.test_envs == [-1]:
            args.test_envs = [env for env in envs if env not in args.train_envs]
            
    print(f'train_envs: {args.train_envs}, test_envs: {args.test_envs}')

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
        
    print("Without Data Augmentation!!!")
    hparams['data_augmentation'] = False

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs + args.train_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]
    


    num_workers_eval = 8 if args.dataset != "DomainNet" else 8
    batch_size_eval = 128 if args.dataset != "DomainNet" else 128
    
    src_test_loaders = [FastDataLoader(
            dataset=env,
            batch_size=num_workers_eval,
            num_workers=num_workers_eval)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.train_envs]
    
    
    src_train_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits)
            if i in args.train_envs]
    
    trg_test_loader = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.test_envs]
    
    # trg_train_loader = [FastDataLoader(
    #         dataset=env,
    #         batch_size=hparams['batch_size'],
    #         num_workers=dataset.N_WORKERS)
    #         for i, (env, env_weights) in enumerate(in_splits)
    #         if i in args.test_envs][0]
    
    eval_class = FastDataLoader if args.dataset != "DomainNet" else dataloader
    eval_loaders = [eval_class(
        dataset=env,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    
    tta = algorithms.PromptTTA(dataset.input_shape, dataset.num_classes,
                               1, hparams)
    
    
    tta.to(device)
    
    print('===='*10+'START'+'===='*10)
    if args.restore:
        ckpt = torch.load(args.restore)
        missing_keys, unexpected_keys = tta.load_state_dict(ckpt["model_dict"], strict=False)
        print("restored from {}".format(args.restore))
        print("missing keys: {}".format(missing_keys))
        print("unexpected keys: {}".format(unexpected_keys))
        
    # domain mapping
    cnt = 0
    domain_mapping = {x: None for x in args.test_envs}
    # for i in range(len(in_splits)):
    #     if i not in args.test_envs:
    #         domain_mapping[i] = cnt
    #         cnt += 1

    src_features, src_labels = [], []
    # for i, loader in enumerate(src_train_loaders):
    print('Using src val!!!')
    for i, loader in enumerate(src_test_loaders):
        features,  labels = misc.forward_raw(tta, loader, device)
        src_features.append(features)
        src_labels.append(labels)
    
    src_features = torch.cat(src_features, dim=0)
    src_labels = torch.cat(src_labels, dim=0)
    print(f'src_features: {src_features.shape}, src_labels: {src_labels.shape}')
    
    # trg_features_before,  trg_labels_before = misc.forward_raw(tta, trg_test_loader, device)
    
    # print(f'trg_features: {trg_features_before.shape}, trg_labels: {trg_labels_before.shape}')
    
    
    # NORM = True
    # LAMBDA = 1e8
    # PRIOR = 'uniform'
    # print(f'WD CONFIG: NORM: {NORM}, LAMBDA: {LAMBDA}, PRIOR: {PRIOR}')

    # curr_wd = misc.compute_labeled_wd2(trg_features_before, trg_labels_before, src_features, src_labels, normalize=NORM, device=device, prior=PRIOR, LAMBDA=LAMBDA)
    # print(f'WD2 before prompt: {curr_wd}')
    
    print()
    # Using TRG test
    # trg_test_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(out_splits)      
    #     if i in args.test_envs[:1]]

    # Using all TRG data
    env_list = []
    # for i, (env, env_weights) in enumerate(in_splits):      
    #     if i in args.test_envs[:1]:
    #         env_list += env
    #         print(f'TRG env: {i}', end='\t')
            
    # print(f'num of data (train set): {len(env_list)}')
    
    for i, (env, env_weights) in enumerate(out_splits):   
        if i in args.test_envs:
            env_list += env
            # print(f'TRG env: {i}', end='\t')
    # print(f'num of data (trg test only): {len(env_list)}')      
    print(f'SRC: {args.train_envs}, TRG: {args.test_envs}')
     
    trg_test_loaders = [InfiniteDataLoader(
        dataset=env_list,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        ]
    
    
    
    trg_test_iterator = zip(*trg_test_loaders)
    
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = len(env_list)/hparams['batch_size'] #min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    checkpoint_freq = len(env_list) // hparams['batch_size'] * 5
    n_steps = int(len(env_list)/hparams['batch_size'] * 5) + 1
    # n_steps = int(len(env_list)/hparams['batch_size'] * 5) + 1 #min((checkpoint_freq + 1) * 10, 31)
    
    change_lr_step = max(5, checkpoint_freq)

    # n_steps = args.steps or dataset.N_STEPS
    # checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            # "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": tta.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    best_acc =  0
    leading_acc = 0
    best_loss = 1e10
    start_time = time.time()
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(trg_test_iterator)]
        
        step_vals = tta.update(minibatches_device, (src_features, src_labels))
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
            
        # if (step % (checkpoint_freq * 3) == 0) and (step != 0):
        # if step == change_lr_step:
        #     # hparams['lr_prompt'] /= 10
        #     tta.prompt_opt.param_groups[0]['lr'] /= 10
        #     print(f'Step {step} Prompt lr: {tta.prompt_opt.param_groups[0]["lr"]}')
            # print(f'Prompt lr: {hparams["lr_prompt"]}')

        # if (step % checkpoint_freq == 0) or (step == n_steps - 1):
        if (step == n_steps - 1):
            
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(tta, loader, weights, device, name=name, domain=None)
                results[name+'_acc'] = acc
                
                # if name == f'env{args.test_envs[0]}_out':
                #     if acc > best_acc:
                #         best_acc = acc
                #         # save_checkpoint(f'prompt_best_acc.pkl')
                #     if results['loss'] < best_loss:
                #         best_loss = results['loss']
                #         leading_acc = acc
                        # save_checkpoint(f'prompt_best_loss.pkl')
                
                

            # results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            
            # add wd2 to log
            # trg_features_raw, trg_features_prompt, trg_labels = misc.forward_prompt_raw_all(tta, trg_test_loader, device, show_acc=False)
            # src_trg_prompt_wd2 = misc.compute_labeled_wd2(trg_features_prompt, trg_labels, src_features, src_labels, normalize=NORM, device=device, prior=PRIOR, LAMBDA=LAMBDA)
            # results['wd2'] = src_trg_prompt_wd2
            
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # algorithm_dict = tta.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}_{hparams["lr_prompt"]}.pkl')
            

    save_checkpoint(f'prompt.pkl')

    # with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    #     f.write('done')
    print()
    print(f'SRC: {args.train_envs}, TRG: {args.test_envs}')
    print(f'Time: {time.time() - start_time:.1f}s')
    # print(f'Best acc: {best_acc:0.3f}, leading acc: {leading_acc:0.3f}, last acc: {results[f"env{args.test_envs[0]}_out_acc"]:0.3f}')
    
    print('===='*10+'END'+'===='*10)
    os._exit(0)
