"""A modified timm training pipeline using Pytorch Lightning. 

Takes a standard timm image model and the standard timm training script and adapts it for the optimizations provided by lightning. Many of the original optimizations are easy to integrate, however there are some that are unsupported by this new script:


- No EMA training
- We do not use timm's distributed logic, instead letting pytorch handle this
- We do not use timm's checkpointing or resuming facilities
- We do not use timm's logging
- We do not keep track of the best metric.
- We do not (distributed) batchnorms (but transformers don't use this)
- We do not handle second order optimizers
- We do not fully handle `args.channels_last` -- we always assume the channels are first
- We must use no_prefetcher as the argument
- Following timm convention, learning rate scheduler is iterated every epoch, but only after validation
- No WandB support
"""

import timm
import torch
from torch import nn

import os
import yaml
import argparse
import functools as ft
import pytorch_lightning as pl
import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data.constants import *
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb');

parser = pl.Trainer.add_argparse_args(parser)

def _parse_args(spec_args=None):
    """Handle arguments if a config file is also passed
    
    `spec_args` allows a list of argument (pairs) to be assigned to args from a notebook
    """
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    if spec_args is not None:
        args = parser.parse_args(args=spec_args)
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

# myargs = [
#     "/gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k",
#     "--cutmix", "1.",
#     "--mixup", "0.8",
#     "--reprob", "0.5",
#     "--aa", "rand",
#     "--train-split", "train",
#     "--val-split", "val",
#     "--batch-size", "12",
#     "--no-prefetcher",
#     "--pin-mem",
#     "--num-classes", "1000",
#     "--model", "et_base_patch16_224",
# ]
# args, args_text = _parse_args(myargs)

def get_dataset_train(args):
    return create_dataset(
    args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
    class_map=args.class_map,
    download=args.dataset_download,
    batch_size=args.batch_size,
    repeats=args.epoch_repeats)

def get_dataset_eval(args):
     return create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

class LitTimm(pl.LightningModule):
    """A simple wrapper around all the components of timm's train.py for classification"""
    def __init__(
        self,
        args, # Config args from timm
    ):
        super().__init__()
        self.model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint)
        self.args = self.proc_args(args) # Call after model created
        # enable split bn (separate bn stats per batch-portion)
        if self.args.split_bn:
            assert self.args.num_aug_splits > 1 or args.resplit
            self.model = convert_splitbn_model(self.model, max(self.args.num_aug_splits, 2))
        
        if self.args.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self._metrics_logged = False
        self.save_hyperparameters()
        
        # Defaults
        self.collate_fn = None
        self.ds_train = None
        self.ds_val = None
        self.validate_loss_fn = nn.CrossEntropyLoss()
        self.train_loss_fn = self.get_train_loss_fn()
        
    def proc_args(self, args):
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
        if args.grad_checkpointing:
            self.model.set_grad_checkpointing(enable=True)
        data_config = resolve_data_config(vars(args), model=self.model, verbose=True)
        args.prefetcher = not args.no_prefetcher
        assert args.no_prefetcher, "Prefetcher case unhandled"
        # setup augmentation batch splits for contrastive loss or split bn
        num_aug_splits = 0
        if args.aug_splits > 0:
            assert args.aug_splits > 1, 'A split of 1 makes no sense'
            num_aug_splits = args.aug_splits
        args.num_aug_splits = num_aug_splits

        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            args.train_interpolation = data_config['interpolation']

        args.distributed = False
        # if 'WORLD_SIZE' in os.environ:
        #     args.distributed = int(os.environ['WORLD_SIZE']) > 1
        # args.device = 'cuda:0'
        # args.world_size = 1
        # args.rank = 0  # global rank

        # setup mixup / cutmix
        self.collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                self.collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
            args.prefetcher = not args.no_prefetcher
            args.distributed = False
            return args
        
        self.data_config = resolve_data_config(vars(args), model=self.model, verbose=args.local_rank == 0)
        
        # setup mixup / cutmix
        self.mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        mixup_fn = None
        if self.mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                self.collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)
        self.mixup_fn = mixup_fn
        return args
        
    def get_train_loss_fn(self):
        args = self.args
        if args.jsd_loss:
            assert args.num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=args.num_aug_splits, smoothing=args.smoothing)
        elif self.mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        return train_loss_fn


    def configure_optimizers(self):
        # Might need to consider alphas here
        args = self.args
        params = [v for k, v in self.model.named_parameters() if "betas" not in k]
        optimizer = create_optimizer_v2(self.model, **optimizer_kwargs(cfg=args))
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        
        self.second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=lr_scheduler,
                interval="epoch",
                frequency=1,
            ),
        )
    
    def setup(self, stage=None):
        # stage = "fit", "test", "validate", "predict"
        # Move datasets here, but debugging will be faster if we don't have to call setup every time
        pass
    
    def train_dataloader(self):
        args = self.args
        data_config = self.data_config

        loader_train = create_loader(
            get_dataset_train(args),
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_repeats=args.aug_repeats,
            num_aug_splits=args.num_aug_splits,
            interpolation=args.train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            # distributed=args.distributed, # WARNING. I think pytorch lightning handles this for me
            collate_fn=self.collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            worker_seeding=args.worker_seeding,
        )
        return loader_train
    
    def val_dataloader(self):
        args = self.args
        data_config = self.data_config
        
        loader_eval = create_loader(
            get_dataset_eval(args),
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            # distributed=args.distributed, # WARNING. I think pytorch lightning handles all this for me
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )
        
        return loader_eval
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        pass
        # I will call this manually in `validation_epoch_end`
        # scheduler.step(epoch=self.current_epoch, metric=metric[self.args.eval_metric])  # timm's scheduler need the epoch value
        
    def training_step(self, batch, batch_idx):

        # My setup looks ok, but how does the training step look?
        input, target = batch
        args = self.args
        if args.mixup_off_epoch and self.current_epoch >= self.args.mixup_off_epoch:
            # if args.prefetcher and loader.mixup_enabled: # UNHANDLED
            #     loader.mixup_enabled = False
            if self.mixup_fn is not None:
                self.mixup_fn.mixup_enabled = False       

        if self.mixup_fn is not None:
            input, target = self.mixup_fn(input, target)

        output = self.model(input)
        loss = self.train_loss_fn(output, target)
        
        return {
            "loss": loss,
            "preds": output,
            "target": torch.argmax(target, dim=-1)
        }
    
    def training_epoch_end(self, train_step_outputs):
        acc1, acc5 = [],[]
        for x in train_step_outputs:
            a1, a5 = accuracy(x['preds'], x['target'], topk=(1,5))
            acc1.append(a1)
            acc5.append(a5)
        
        acc1 = torch.mean(torch.stack(acc1))
        acc5 = torch.mean(torch.stack(acc5))
        self.log("train_acc1", acc1)
        self.log("train_acc5", acc5)
    
    def validation_step(self, batch, batch_idx):
        args = self.args
        with torch.no_grad():
            input, target = batch
            output = self.model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
                
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
                
            # Collect these metrics on validation end
            loss = self.validate_loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5)) #timm.utils.metrics
            
            return {
                "loss": loss,
                "acc1": acc1,
                "acc5": acc5
            }
        
    def validation_step_end(self, batch_parts):
        try:
            acc1 = torch.mean(torch.stack(batch_parts["acc1"]))
            acc5 = torch.mean(torch.stack(batch_parts["acc5"]))
            loss = torch.mean(torch.stack(batch_parts["loss"]))
        except TypeError:
            # We are dealing with a single tensor and single device
            acc1 = batch_parts["acc1"]
            acc5 = batch_parts["acc5"] 
            loss = batch_parts["loss"]
            
        self.log("val_acc1", acc1)
        self.log("val_acc5", acc5)
            
        return {
            "loss": loss,
            "acc1": acc1,
            "acc5": acc5
        }
    
    def validation_epoch_end(self, val_step_outputs):
        acc1 = torch.mean(torch.stack([v["acc1"] for v in val_step_outputs]))
        acc5 = torch.mean(torch.stack([v["acc5"] for v in val_step_outputs]))
        loss = torch.mean(torch.stack([v["loss"] for v in val_step_outputs]))
        
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step(epoch=self.current_epoch + 1, metric=acc1)

        out = {
            "loss": loss,
            "acc1": acc1,
            "acc5": acc5
        }
        self.log("val_loss", loss)
        self.log("val_acc1", acc1)
        self.log("val_acc5", acc5)
        
if __name__ == "__main__":
    args, args_text = _parse_args()
    litmodel = LitTimm(args)
    trainer = pl.Trainer.from_argparse_args(args, overfit_batches=100)
    # trainer = pl.Trainer(accelerator="gpu", devices=1, overfit_batches=10000, grad_clip_val=0.5) # Test model
    trainer.fit(litmodel)
    print("Done")