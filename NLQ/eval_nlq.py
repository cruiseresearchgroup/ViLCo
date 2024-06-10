import argparse
import os
import time
import torch
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, ReferringRecall, valid_one_epoch_nlq_singlegpu
from libs.datasets.data_utils import trivial_batch_collator


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    print(args.config)
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=trivial_batch_collator,
        batch_size=16,
        num_workers=8,
        shuffle=False,
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    """4. load ckpt"""

    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(args.resume, map_location="cpu")
    # args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    # also load the optimizer / scheduler if necessary
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
        args.resume, checkpoint['epoch']
    ))
    model.to(torch.device("cuda:0"))

    # set up evaluator
    det_eval = ReferringRecall(dataset=cfg["dataset_name"],gt_file=cfg["dataset"]["json_file"])

    output_file = None
    if args.save:
        output_file = os.path.join(os.path.split(args.resume)[0], 'nlq_predictions_epoch_val_top10_%d.json'%checkpoint['epoch'])

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    results = valid_one_epoch_nlq_singlegpu(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('resume', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--save', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
