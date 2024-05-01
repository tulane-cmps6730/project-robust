

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
import re

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
    parser.add_argument('--train_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--restore_prompt', type=str, default=None)
    args = parser.parse_args()
    
    if args.dataset == 'PACS' or args.dataset == 'VLCS' or args.dataset == 'OfficeHome' or args.dataset == 'TerraIncognita':
        envs = [0, 1, 2, 3]
        if args.train_envs == [-1]:
            args.train_envs = [env for env in envs if env not in args.test_envs]
        if args.test_envs == [-1]:
            args.test_envs = [env for env in envs if env not in args.train_envs]

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
    

    # uda_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_splits)
    #     if i in args.test_envs]

    num_workers_eval = 8 if args.dataset != "DomainNet" else 8
    batch_size_eval = 128 if args.dataset != "DomainNet" else 128
    
    src_test_loaders = [FastDataLoader(
            dataset=env,
            batch_size=num_workers_eval,
            num_workers=num_workers_eval)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.train_envs]
    
    # hparams['batch_size'] = 32
    
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
            if i in args.test_envs][0]
    
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

    # algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    # algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
    #     len(args.train_envs), hparams)

    # if algorithm_dict is not None:
    #     algorithm.load_state_dict(algorithm_dict)

    # algorithm.to(device)
    
    tta = algorithms.PromptTTA(dataset.input_shape, dataset.num_classes,
                               1, hparams)
    
    
    tta.to(device)
    # if args.restore:
    #     ckpt = torch.load(args.restore)
    #     missing_keys, unexpected_keys = algorithm.load_state_dict(ckpt["model_dict"])
        
    #     print("restored from {}".format(args.restore))
    #     print("missing keys: {}".format(missing_keys))
    #     print("unexpected keys: {}".format(unexpected_keys))
    
    print('===='*10+'START'+'===='*10)
    if args.restore_prompt:
        ckpt = torch.load(args.restore_prompt)
        msg = tta.load_state_dict(ckpt["model_dict"])
        print(f"restored from {args.restore_prompt} with msg: {msg}")
        
    
    # domain mapping
    cnt = 0
    domain_mapping = {x: None for x in args.test_envs}
    # for i in range(len(in_splits)):
    #     if i not in args.test_envs:
    #         domain_mapping[i] = cnt
    #         cnt += 1

    # src_test_iterator = zip(*src_test_loaders)
    
    # src_test_list = [(x.to(device), y.to(device))
    #         for x,y in next(src_test_iterator)]
    src_features, src_labels = [], []
    for i, loader in enumerate(src_test_loaders):
        # features,  labels = misc.forward_raw(algorithm, loader, device)
        features,  labels = misc.forward_raw(tta, loader, device)
        src_features.append(features)
        src_labels.append(labels)
    
    trg_features_raw, trg_features_prompt, trg_labels, ent, ent_prompt = misc.forward_prompt_raw_all(tta, trg_test_loader, device)
    print(f'trg_features_raw: {trg_features_raw.shape}, trg_features_prompt: {trg_features_prompt.shape}, trg_labels: {trg_labels.shape}')
    
    src_features = torch.cat(src_features, dim=0)
    src_labels = torch.cat(src_labels, dim=0)
    print(f'src_features: {src_features.shape}, src_labels: {src_labels.shape}')
    
    # print("random sampling")
    # idx = np.random.choice(src_features.shape[0], len(trg_labels), replace=False)
    # src_features = src_features[idx]
    # src_labels = src_labels[idx]
    
    print(f'src_features: {src_features.shape}, src_labels: {src_labels.shape}')
    reps = {'src_features': src_features, 'src_labels': src_labels, 'trg_features_raw': trg_features_raw, 'trg_features_prompt': trg_features_prompt, 'trg_labels': trg_labels}
    
    torch.save(reps, '/media/zybeich/prompt_tta/plots/pacs_p2s_reps.pt')
    os._exit(0) 
    
    NORM = True
    LAMBDA = 1e8
    PRIOR = 'uniform'
    print(f'WD CONFIG: NORM: {NORM}, LAMBDA: {LAMBDA}, PRIOR: {PRIOR}')
    src_trg_wd2 = misc.compute_labeled_wd2(trg_features_raw, trg_labels, src_features, src_labels, normalize=NORM, device=device, prior=PRIOR, LAMBDA=LAMBDA)
    src_trg_prompt_wd2 = misc.compute_labeled_wd2(trg_features_prompt, trg_labels, src_features, src_labels, normalize=NORM, device=device, prior=PRIOR, LAMBDA=LAMBDA)
    
    wd2 = misc.compute_wd2(trg_features_raw, src_features, device, normalize=NORM, LAMBDA=LAMBDA)
    wd2_prompt = misc.compute_wd2(trg_features_prompt, src_features, device, normalize=NORM, LAMBDA=LAMBDA)
    
    # trg_trg_prompt_wd2 = misc.compute_labeled_wd2(trg_features_raw, trg_labels, trg_features_prompt, trg_labels, normalize=NORM, device=device, prior=PRIOR, LAMBDA=LAMBDA)
    
    # print(f'Before prompt: {src_trg_wd2:.2f}')
    # print(f'After prompt: {src_trg_prompt_wd2:.2f}')
    # print(f'Prompt vs Raw: {trg_trg_prompt_wd2:.2f}')
    
    print(f'{src_trg_wd2:.2f}, {src_trg_prompt_wd2:.2f}, {wd2:.2f}, {wd2_prompt:.2f}, {ent:.2f}, {ent_prompt:.2f}')
    
    modes = ['all', 'only_trg', 'src_trg_before', 'src_trg_after']
    # step = re.findall(r'step\d+', os.path.basename(args.restore_prompt))[0]
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    
    feats_list = [src_features, trg_features_raw, trg_features_prompt]
    labels_list = [src_labels, trg_labels, trg_labels]
    step_list = ['src', 'trg_before', 'trg_after']
    
    for mode in modes:
        filename = f'tsne_{mode}.pdf'
        misc.plot_tsne(feats_list, labels_list, step_list, os.path.join(args.output_dir, filename), mode=mode)
        
    print('===='*10+'END'+'===='*10)
    print()
    torch.cuda.empty_cache()
    os._exit(0) 
    # sys.exit()
    