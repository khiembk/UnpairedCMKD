import os
import sys
import torch
import argparse
import numpy as np
import random
import wandb
from utils.helper import gen_data, gen_data_challenging, train_network_distill, train_network_distill_unpair_sumall, train_network_distill_unpair_ce, train_network_distill_unpair_bilevel, train_network_distill_unpair_vanillaKD, \
     train_network_distill_unpair_fea, pre_train, train_network_distill_unpair_bilevel_with_different_metric
# from utils.model import ImageNet, AudioNet
from utils.model_res import ImageNet, AudioNet
from utils.module import Tea, Stu
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval_overlap_tag(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    # load teacher model
    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    if stu_type == 1:
        arch = args.image_arch
        print(f'teacher:image ({args.image_arch}); student:audio({args.audio_arch})')
    elif stu_type == 0:
        arch = args.audio_arch
        print(f'teacher:audio ({args.audio_arch}); student:image({args.image_arch})')
    tea_model.load_state_dict(
        torch.load('/ckpts/ravvdess_ckpts/teacher_mod_' + str(tea_type) + '_' + arch + '_' + str(args.num_frame) + '_overlap.pkl',
                   map_location={"cuda:0": "cpu"}), strict=False)
    print(f'Finish Loading teacher model')

    net = ImageNet(args).to(device) if stu_type == 0 else AudioNet(args).to(device)
    tea = Tea().cuda()
    stu = Stu().cuda()
    optimizer = torch.optim.SGD([
            {'params': net.parameters()},
            {'params': tea_model.parameters()},
            {'params': tea.parameters()},
            {'params': stu.parameters()},
        ], lr=args.lr, momentum=0.9)
    
    optimizer2 = torch.optim.Adam([
            {'params': net.parameters()},
            {'params': tea_model.parameters()},
            {'params': tea.parameters()},
            {'params': stu.parameters()},
        ],
        lr = args.lr, 
        betas = (args.beta1, args.beta2),
        weight_decay=args.weight_decay
        )
    
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.min_lr / args.lr, end_factor=1.0, total_iters=args.warmup_epoch)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epoch, eta_min=args.min_lr)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_epoch])
    

    if args.method_type == "cost_metric":
        cost_metric = args.cost_metric
        acc = train_network_distill_unpair_bilevel_with_different_metric(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu, metric= cost_metric)
    if args.method_type == "bilevel":
        transfer_kernel = args.transfer_kernel
        acc = train_network_distill_unpair_bilevel(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu, transfer_kernel)
    if args.method_type == "ce":
        acc = train_network_distill_unpair_ce(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu)
    if args.method_type == "sumall":
        acc = train_network_distill_unpair_sumall(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu)    
    if args.method_type == "vanillaKD":
        acc = train_network_distill_unpair_vanillaKD(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu)
    if args.method_type =="feadistill":
        acc = train_network_distill_unpair_fea(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu)
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--stu-type', type=int, default=0, help='the modality of student unimodal network, 0 for image, 1 for audio')
    parser.add_argument('--num-runs', type=int, default=1, help='num runs')
    parser.add_argument('--num-epochs', type=int, default=100, help='num epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--batch-size2', type=int, default=512, help='batch size for calculating the overlap tag')
    parser.add_argument('--num-workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-2, help='lr')
    parser.add_argument('--num_frame', type=int, default=1)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--audio_arch', type=str, default='resnet18')
    parser.add_argument('--image_arch', type=str, default='resnet18')
    parser.add_argument('--krc', type=float, default=0.0)
    parser.add_argument('--pre_train', type=int, default=0, help='pre_train student and teacher models')
    parser.add_argument('--cmkd', type=int, default=1, help='crossmodal knowledge distillation')
    parser.add_argument('--group', type=str, default='c2kd', help='group of experiments')
    # add
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)
    parser.add_argument("--method_type", type=str, default="ce")
    parser.add_argument("--cost_metric", type=str, default="chordal")
    parser.add_argument("--transfer_kernel", type=int, default=2)

    # --- Challenging unpaired settings ---
    parser.add_argument("--challenge", action="store_true", default=False,
                        help="Use challenging unpaired dataloader")
    parser.add_argument("--challenge_preset", type=str, default="moderate",
                        choices=["clean", "mild", "moderate", "hard",
                                 "marginal_only", "domain_only", "imbalance_only"],
                        help="Difficulty preset for challenging dataloader")
    parser.add_argument("--marginal_mismatch", action="store_true", default=False,
                        help="Enable disjoint audio/image sample pools per class")
    parser.add_argument("--marginal_ratio", type=float, default=0.5,
                        help="Fraction of class samples in audio pool (rest in image)")
    parser.add_argument("--domain_shift", action="store_true", default=False,
                        help="Enable domain-shift augmentations")
    parser.add_argument("--domain_shift_level", type=float, default=0.5,
                        help="Severity of domain shift [0=mild, 1=heavy]")
    parser.add_argument("--label_imbalance", action="store_true", default=False,
                        help="Enable long-tail label distribution for one modality")
    parser.add_argument("--imbalance_modality", type=str, default="audio",
                        choices=["audio", "image"],
                        help="Which modality gets the imbalanced view")
    parser.add_argument("--imbalance_factor", type=float, default=10.0,
                        help="Max/min class ratio for label imbalance")

    args = parser.parse_args()

    wandb.login(key="YOUR_KEY")


    print(args)

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    data_dir = '/ravvdess'
    if args.challenge:
        loader = gen_data_challenging(data_dir, args.batch_size, args.num_workers, args)
    else:
        loader = gen_data(data_dir, args.batch_size, args.num_workers, args)
    if args.pre_train:
        loader_fb = gen_data(data_dir, args.batch_size2, args.num_workers, args)
        pre_train(args.stu_type, loader, args.num_epochs, args.lr, device, args)

    if args.cmkd:
        log_np = np.zeros((args.num_runs, 4))
        for run in range(args.num_runs):
            set_random_seed(seed=run)
            print(f'Seed {run}')

            run_name = f"data: ravdess, type:unpair, {args.method_type}, seed: {run} lr_{args.lr}_bs_{args.batch_size}_numepochs_{args.num_epochs}_stutype_{args.stu_type}"
            wandb.init(
                entity="cmkd",
                project="experiments",
                name=run_name,
                config=vars(args),
                group=args.group,  
                reinit=True        
            )

            log_np[run, :] = eval_overlap_tag(loader, device, args)

            wandb.finish()
        log_mean = np.mean(log_np, axis=0)
        log_std = np.std(log_np, axis=0)
        print(f'Finish {args.num_runs} runs')
        print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')
        print(f'Teacher Val Acc {log_mean[2]:.3f} ± {log_std[2]:.3f} | Test Acc {log_mean[3]:.3f} ± {log_std[3]:.3f}')
        if args.stu_type == 1:
            arch = args.image_arch
            print(f'teacher:image ({args.image_arch}); student:audio({args.audio_arch})')
        elif args.stu_type == 0:
            arch = args.audio_arch
            print(f'teacher:audio ({args.audio_arch}); student:image({args.image_arch})')