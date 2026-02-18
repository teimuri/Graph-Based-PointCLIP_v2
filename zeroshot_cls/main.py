import os
import torch
import argparse
from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import datasets.scanobjnn
import datasets.modelnet40
from trainers import best_param

from trainers import zeroshot
from trainers.post_search import search_weights_zs, search_prompt_zs

import torchvision.transforms as T


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.gnn_dir:
        cfg.GNN_DIR = args.gnn_dir

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.TRAINER.EXTRA = CN()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup_cfg(args)
    
    # set random seed
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)

    # 1. ZERO-SHOT MODE
    if args.zero_shot:
        if args.gnn_dir:
            trainer.gnn_aggregator.load_gnn(args.gnn_dir)
        trainer.test_zs()
        
        if args.post_search:
            vweights = best_param.best_prompt_weight['{}_{}_test_weights'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
            prompts = best_param.best_prompt_weight['{}_{}_test_prompts'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
        
            prompts, image_feature = search_prompt_zs(cfg, vweights, searched_prompt=prompts)
            return

    # 2. STANDARD TRAINING MODE
    elif not args.no_train:
        print("Starting custom PyTorch training loop...")
        
        train_loader = trainer.train_loader_x
        
        # Determine the validation loader (Dassl usually provides val_loader or a test_loader dict)
        if hasattr(trainer, 'val_loader') and trainer.val_loader is not None:
            val_loader = trainer.val_loader
        elif hasattr(trainer, 'test_loader'):
            # Dassl test_loader is often a dict containing dataset names as keys
            val_loader = list(trainer.test_loader.values())[0] if isinstance(trainer.test_loader, dict) else trainer.test_loader
        else:
            val_loader = None
            print("Warning: No validation or test loader found.")
        max_epochs = cfg.OPTIM.MAX_EPOCH
        best_val_acc = 0.0  # Keep track of the best accuracy

        for epoch in range(max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
            
            # --- TRAINING PHASE ---
            trainer.model.train()
            for batch_idx, batch in enumerate(train_loader):
                
                loss_summary = trainer.forward_backward(batch, training=True)
                
                if batch_idx % 10 == 0:
                    loss = loss_summary["loss"]
                    ce_loss = loss_summary["ce_loss"]
                    supcon_loss = loss_summary["supcon_loss"]
                    acc = loss_summary["acc"]
                    current_lr = trainer.optim.param_groups[0]['lr']
                    print(f"Train Batch {batch_idx} | LR: {current_lr:.6f} | Loss: {loss:.4f} | ce_loss: {ce_loss:.3f} | con_loss: {supcon_loss:.3f} | Acc: {acc:.2f}%")
            
            if trainer.sched is not None:
                trainer.sched.step()

            # --- VALIDATION PHASE ---
            if val_loader is not None:
                raise ValueError(val_loader)
                
                trainer.model.eval()
                val_loss, val_ce, val_supcon, val_acc = 0.0, 0.0, 0.0, 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        loss_summary = trainer.forward_backward(batch, training=False)
                        val_loss += loss_summary["loss"]
                        val_ce += loss_summary["ce_loss"]
                        val_supcon += loss_summary["supcon_loss"]
                        val_acc += loss_summary["acc"]
                        num_batches += 1
                
                if num_batches > 0:
                    avg_val_loss = val_loss / num_batches
                    avg_val_acc = val_acc / num_batches
                    print(f"--> Validation | Loss: {avg_val_loss:.4f} | ce_loss: {val_ce/num_batches:.3f} | con_loss: {val_supcon/num_batches:.3f} | Acc: {avg_val_acc:.2f}%")
                    
                    # Optional: Save logic for the best model
                    if avg_val_acc > best_val_acc:
                        best_val_acc = avg_val_acc
                        print(f"🌟 New best validation accuracy: {best_val_acc:.2f}%")
                        # You can trigger your checkpoint saving here if needed

        # Save final GNN weights
        if args.gnn_dir:
            trainer.gnn_aggregator.save_gnn(args.gnn_dir)
        
        # Final Zero-Shot / Eval Test
        trainer.test_zs()
            
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='', help='output directory')
    parser.add_argument('--seed', type=int,default=2,help='only positive value enables a fixed seed')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation methods')
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--dataset-config-file', type=str, default='', help='path to config file for dataset setup')
    parser.add_argument('--trainer', type=str, default='', help='name of trainer')
    parser.add_argument('--backbone', type=str, default='', help='name of CNN backbone')
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument('--zero-shot', action='store_true', help='zero-shot only')
    parser.add_argument('--post-search', default=True, action='store_true', help='post-search only')
    parser.add_argument('--model-dir', type=str, default='',help='load model from this directory for eval-only mode')
    parser.add_argument('--gnn-dir', type=str, default='',help='load gnn from this directory')
    parser.add_argument('--load-epoch', type=int, default=175, help='load model weights at this epoch for evaluation')
    parser.add_argument('--no-train', action='store_true', help='do not call trainer.train()')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='modify config options using the command-line')
    args = parser.parse_args()
    main(args)
    
