import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets import ImageFolder
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.autoencoder import *
from model.video_swin_transformer import *
from utils import *
from model.pseudoanomaly_utils import create_pseudoanomaly_cifar_smooth, \
    create_pseudoanomaly_cifar_smoothborder, create_pseudoanomaly_seq_smoothborder, \
    create_pseudoanomaly_cifar_cutmix, create_pseudoanomaly_cifar_mixupcutmix

import time
from model import EntropyLossEncap

import argparse

parser = argparse.ArgumentParser(description="STEAL Net")
parser.add_argument('--model', type=str, default='VST', choices=['VST','conAE'])
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2','avenue', 'shanghai','SW_video','VAD'], help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')

parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# related to skipping frame pseudo anomaly
parser.add_argument('--pseudo_anomaly_jump_inpainting', type=float, default=0, help='pseudo anomaly jump frame (skip frame) probability but with inpainting-like loss. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[3], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3

# related to patch based pseudo anomaly
parser.add_argument('--pseudo_anomaly_cifar_inpainting_smooth', type=float, default=0, help='pseudo anomaly using cifar100 patch (SmoothMixC) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_shanghai_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using shanghai patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_ped2_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using ped2 patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_SW_video_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using SW_video patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_VAD_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using VAD patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using cifar100 patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_cutmix', type=float, default=0, help='pseudo anomaly using cifar100 patch (CutMix) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size hyperparameter')
parser.add_argument('--pseudo_anomaly_imagenet_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using imagenet patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_mixupcutmix', type=float, default=0, help='pseudo anomaly using cifar100 patch (MixUp-patch) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size hyperparameter')
parser.add_argument('--max_size', type=float, default=0.2, help='maximum size of the patch relative to the input (hyperparameter alpha)')
parser.add_argument('--max_move', type=int, default=0, help='maximum movement in pixel of the patch to the input (hyperparameter beta)')

parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')
parser.add_argument('--Entropy_Loss_Weight', type=float, default=0.0002, help='entropy loss weight')
parser.add_argument('--Period_Loss_Weight', type=float, default=0.02, help='period loss weight')

##################

args = parser.parse_args()
entropy_loss_weight = args.Entropy_Loss_Weight
period_loss_weight = args.Period_Loss_Weight
device = torch.device("cuda")
tr_entropy_loss_func = EntropyLossEncap().to(device)
# assert 1 not in args.jump

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += '_'
exp_dir += args.model
exp_dir += '_weight'
exp_dir += '_recon_256'

exp_dir += '_pajumpin' + str(args.pseudo_anomaly_jump_inpainting) if args.pseudo_anomaly_jump_inpainting != 0 else ''
exp_dir += '_jump[' + ','.join([str(args.jump[i]) for i in range(0,len(args.jump))]) + ']' if args.pseudo_anomaly_jump_inpainting != 0 else ''

exp_dir += '_pacifinS' + str(args.pseudo_anomaly_cifar_inpainting_smooth) if args.pseudo_anomaly_cifar_inpainting_smooth != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_cifar_inpainting_smooth != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_cifar_inpainting_smooth != 0 and args.max_move > 0 else ''

exp_dir += '_pacifinSB' + str(args.pseudo_anomaly_cifar_inpainting_smoothborder) if args.pseudo_anomaly_cifar_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_cifar_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_cifar_inpainting_smoothborder != 0 and args.max_move > 0 else ''

exp_dir += '_pacifinC' + str(args.pseudo_anomaly_cifar_inpainting_cutmix) if args.pseudo_anomaly_cifar_inpainting_cutmix != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_cifar_inpainting_cutmix != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_cifar_inpainting_cutmix != 0 and args.max_move > 0 else ''

exp_dir += '_pacifinMC' + str(args.pseudo_anomaly_cifar_inpainting_mixupcutmix) if args.pseudo_anomaly_cifar_inpainting_mixupcutmix != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_cifar_inpainting_mixupcutmix != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_cifar_inpainting_mixupcutmix != 0 and args.max_move > 0 else ''

exp_dir += '_papedinSB' + str(args.pseudo_anomaly_ped2_inpainting_smoothborder) if args.pseudo_anomaly_ped2_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_ped2_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_ped2_inpainting_smoothborder != 0 and args.max_move > 0 else ''

exp_dir += '_papedinSB' + str(args.pseudo_anomaly_SW_video_inpainting_smoothborder) if args.pseudo_anomaly_SW_video_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_SW_video_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_SW_video_inpainting_smoothborder != 0 and args.max_move > 0 else ''

exp_dir += '_papedinSB' + str(args.pseudo_anomaly_VAD_inpainting_smoothborder) if args.pseudo_anomaly_VAD_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_VAD_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_VAD_inpainting_smoothborder != 0 and args.max_move > 0 else ''

exp_dir += '_pashinSB' + str(args.pseudo_anomaly_shanghai_inpainting_smoothborder) if args.pseudo_anomaly_shanghai_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_shanghai_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_shanghai_inpainting_smoothborder != 0 and args.max_move > 0 else ''

exp_dir += '_paimginSB' + str(args.pseudo_anomaly_imagenet_inpainting_smoothborder) if args.pseudo_anomaly_imagenet_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_size) if args.pseudo_anomaly_imagenet_inpainting_smoothborder != 0 else ''
exp_dir += '-' + str(args.max_move) if args.pseudo_anomaly_imagenet_inpainting_smoothborder != 0 and args.max_move > 0 else ''

print('exp_dir: ', exp_dir)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, jump=args.jump, return_normal_seq=args.pseudo_anomaly_jump_inpainting > 0, img_extension=img_extension)

# if args.pseudo_anomaly_cifar_inpainting_smooth > 0 or args.pseudo_anomaly_cifar_inpainting_smoothborder > 0 or args.pseudo_anomaly_cifar_inpainting_cutmix > 0 or args.pseudo_anomaly_cifar_inpainting_mixupcutmix > 0 :
#     # cifar_transform = transforms.Compose([
#     #             transforms.RandomCrop(32, padding=12, padding_mode='reflect'),
#     #             transforms.RandomHorizontalFlip(),
#     #             transforms.RandomVerticalFlip(),
#     #             transforms.ToTensor()
#     # ])
#     cifar_dataset = CIFAR100('dataset/cifar100', transform=transforms.ToTensor(), download=True)
#     cifar_batch = data.DataLoader(cifar_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                   drop_last=True)
#     cifar_iter = iter(cifar_batch)

# if args.pseudo_anomaly_shanghai_inpainting_smoothborder:
#     shanghai_folder = os.path.join(args.dataset_path, 'shanghai', 'training', 'frames')
#     shanghai_dataset = Reconstruction3DDataLoader(shanghai_folder, transforms.Compose([transforms.ToTensor()]),
#                                                resize_height=args.h, resize_width=args.w, dataset='shanghai',
#                                                img_extension=img_extension)


#     shanghai_batch = data.DataLoader(shanghai_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                      drop_last=True)
#     shanghai_iter = iter(shanghai_batch)

# if args.pseudo_anomaly_ped2_inpainting_smoothborder:
#     ped2_folder = os.path.join(args.dataset_path, 'ped2', 'training', 'frames')
#     ped2_dataset = Reconstruction3DDataLoader(ped2_folder, transforms.Compose([transforms.ToTensor()]),
#                                                resize_height=args.h, resize_width=args.w, dataset='ped2',
#                                                img_extension=img_extension)


#     ped2_batch = data.DataLoader(ped2_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                      drop_last=True)
#     ped2_iter = iter(ped2_batch)

# if args.pseudo_anomaly_SW_video_inpainting_smoothborder:
#     SW_video_folder = os.path.join(args.dataset_path, 'SW_video', 'training', 'frames')
#     SW_video_dataset = Reconstruction3DDataLoader(SW_video_folder, transforms.Compose([transforms.ToTensor()]),
#                                                resize_height=args.h, resize_width=args.w, dataset='SW_video',
#                                                img_extension=img_extension)


#     SW_video_batch = data.DataLoader(SW_video_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                      drop_last=True)
#     SW_video_iter = iter(SW_video_batch)

# if args.pseudo_anomaly_VAD_inpainting_smoothborder:
#     VAD_folder = os.path.join(args.dataset_path, 'VAD', 'training', 'frames')
#     VAD_dataset = Reconstruction3DDataLoader(VAD_folder, transforms.Compose([transforms.ToTensor()]),
#                                                resize_height=args.h, resize_width=args.w, dataset='VAD',
#                                                img_extension=img_extension)


#     VAD_batch = data.DataLoader(VAD_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                      drop_last=True)
#     VAD_iter = iter(VAD_batch)

# if args.pseudo_anomaly_imagenet_inpainting_smoothborder > 0:
#     imagenet_transform = transforms.Compose([
#         transforms.RandomResizedCrop((args.h, args.w)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor()
#     ])
#     imagenet_dataset = ImageFolder('dataset/imagenet/train', transform=imagenet_transform)

#     imagenet_batch = data.DataLoader(imagenet_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
#                                      drop_last=True)
#     imagenet_iter = iter(imagenet_batch)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

tic = time.time()

if args.start_epoch < args.epochs:
    if args.model=='VST':
        model = VST()
    else:
        model = convAE()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume
    if args.model_dir is not None:
        assert args.start_epoch > 0
        # Loading the trained model
        model_dict = torch.load(args.model_dir)
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])
        model.cuda()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        print(epoch)
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            #imgs (batch_size,3,16,H,W)
            net_in = copy.deepcopy(imgs['batch'])
            net_in = net_in.cuda()
            img_index = copy.deepcopy(imgs['index'])
            img_index = img_index.cuda()
            # len=batch_size index

            jump_inpainting_pseudo_stat = []
            cifar_inpainting_smooth_pseudo_stat = []
            cifar_inpainting_smoothborder_pseudo_stat = []
            cifar_inpainting_cutmix_pseudo_stat = []
            cifar_inpainting_mixupcutmix_pseudo_stat = []
            ped2_inpainting_smoothborder_pseudo_stat = []
            SW_video_inpainting_smoothborder_pseudo_stat = []
            VAD_inpainting_smoothborder_pseudo_stat = []
            imagenet_inpainting_smoothborder_pseudo_stat = []
            shanghai_inpainting_smoothborder_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False

                # skip frame pseudo anomaly but with inpainting loss
                # pseudo_anomaly_jump_inpainting = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump_inpainting
                # total_pseudo_prob += args.pseudo_anomaly_jump_inpainting
                # if pseudo_anomaly_jump_inpainting:
                #     net_in[b] = imgsjump[0][b].cuda()
                #     jump_inpainting_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     jump_inpainting_pseudo_stat.append(False)

                # # cifar inpainting smooth pseudo anomaly
                # pseudo_anomaly_cifar_inpainting_smooth = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_cifar_inpainting_smooth
                # total_pseudo_prob += args.pseudo_anomaly_cifar_inpainting_smooth
                # if pseudo_anomaly_cifar_inpainting_smooth:
                #     try:
                #         # Samples the batch
                #         cifar_img, _ = next(cifar_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         cifar_iter = iter(cifar_batch)
                #         cifar_img, _ = next(cifar_iter)
                #     net_in[b], mask = create_pseudoanomaly_cifar_smooth(net_in[b], cifar_img[0], args.max_size,
                #                                                         args.h, args.w,
                #                                                         args.dataset_type, max_move=args.max_move)
                #     cifar_inpainting_smooth_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     cifar_inpainting_smooth_pseudo_stat.append(False)

                # # cifar inpainting smooth border pseudo anomaly
                # pseudo_anomaly_cifar_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_cifar_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_cifar_inpainting_smoothborder
                # if pseudo_anomaly_cifar_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         cifar_img, _ = next(cifar_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         cifar_iter = iter(cifar_batch)
                #         cifar_img, _ = next(cifar_iter)
                #     net_in[b], mask = create_pseudoanomaly_cifar_smoothborder(net_in[b], cifar_img[0], args.max_size,
                #                                                               args.h, args.w,
                #                                                               args.dataset_type, max_move=args.max_move)

                #     # imgs_num = (net_in[b, :, 8].cpu().detach().numpy() + 1) * 127.5
                #     # imgs_num = imgs_num.transpose(1, 2, 0).astype(dtype=np.uint8)
                #     # cv2.imshow('a', imgs_num)
                #     # imgs_num = (net_in[b, :, 9].cpu().detach().numpy() + 1) * 127.5
                #     # imgs_num = imgs_num.transpose(1, 2, 0).astype(dtype=np.uint8)
                #     # cv2.imshow('b', imgs_num)
                #     # imgs_num = (net_in[b, :, 10].cpu().detach().numpy() + 1) * 127.5
                #     # imgs_num = imgs_num.transpose(1, 2, 0).astype(dtype=np.uint8)
                #     # cv2.imshow('c', imgs_num)
                #     # cv2.waitKey(0)
                #     cifar_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     cifar_inpainting_smoothborder_pseudo_stat.append(False)

                # # cifar inpainting cutmix pseudo anomaly
                # pseudo_anomaly_cifar_inpainting_cutmix = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_cifar_inpainting_cutmix
                # total_pseudo_prob += args.pseudo_anomaly_cifar_inpainting_cutmix
                # if pseudo_anomaly_cifar_inpainting_cutmix:
                #     try:
                #         # Samples the batch
                #         cifar_img, _ = next(cifar_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         cifar_iter = iter(cifar_batch)
                #         cifar_img, _ = next(cifar_iter)
                #     net_in[b], mask = create_pseudoanomaly_cifar_cutmix(net_in[b], cifar_img[0], args.max_size,
                #                                                         args.h, args.w,
                #                                                         args.dataset_type, max_move=args.max_move)

                #     cifar_inpainting_cutmix_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     cifar_inpainting_cutmix_pseudo_stat.append(False)


                # # cifar inpainting mixupcutmix pseudo anomaly
                # pseudo_anomaly_cifar_inpainting_mixupcutmix = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_cifar_inpainting_mixupcutmix
                # total_pseudo_prob += args.pseudo_anomaly_cifar_inpainting_mixupcutmix
                # if pseudo_anomaly_cifar_inpainting_mixupcutmix:
                #     try:
                #         # Samples the batch
                #         cifar_img, _ = next(cifar_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         cifar_iter = iter(cifar_batch)
                #         cifar_img, _ = next(cifar_iter)
                #     net_in[b], mask = create_pseudoanomaly_cifar_mixupcutmix(net_in[b], cifar_img[0], args.max_size,
                #                                                              args.h, args.w,
                #                                                              args.dataset_type, max_move=args.max_move)
                #     cifar_inpainting_mixupcutmix_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     cifar_inpainting_mixupcutmix_pseudo_stat.append(False)


                # # ped2 inpainting smooth border pseudo anomaly
                # pseudo_anomaly_ped2_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_ped2_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_ped2_inpainting_smoothborder
                # if pseudo_anomaly_ped2_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         ped2_seq = next(ped2_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         ped2_iter = iter(ped2_batch)
                #         ped2_seq = next(ped2_iter)
                #     net_in[b], mask = create_pseudoanomaly_seq_smoothborder(net_in[b], ped2_seq[0], args.max_size,
                #                                                             args.h, args.w,
                #                                                             args.dataset_type, max_move=args.max_move)

                #     ped2_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     ped2_inpainting_smoothborder_pseudo_stat.append(False)

                # # SW_video inpainting smooth border pseudo anomaly
                # pseudo_anomaly_SW_video_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_SW_video_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_SW_video_inpainting_smoothborder
                # if pseudo_anomaly_SW_video_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         SW_video_seq = next(SW_video_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         SW_video_iter = iter(SW_video_batch)
                #         SW_video_seq = next(SW_video_iter)
                #     net_in[b], mask = create_pseudoanomaly_seq_smoothborder(net_in[b], SW_video_seq[0], args.max_size,
                #                                                             args.h, args.w,
                #                                                             args.dataset_type, max_move=args.max_move)

                #     SW_video_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     SW_video_inpainting_smoothborder_pseudo_stat.append(False)

                # # VAD inpainting smooth border pseudo anomaly
                # pseudo_anomaly_VAD_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_VAD_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_VAD_inpainting_smoothborder
                # if pseudo_anomaly_VAD_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         VAD_seq = next(VAD_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         VAD_iter = iter(VAD_batch)
                #         VAD_seq = next(VAD_iter)
                #     net_in[b], mask = create_pseudoanomaly_seq_smoothborder(net_in[b], VAD_seq[0], args.max_size,
                #                                                             args.h, args.w,
                #                                                             args.dataset_type, max_move=args.max_move)

                #     VAD_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     VAD_inpainting_smoothborder_pseudo_stat.append(False)


                # # shanghai inpainting smooth border pseudo anomaly
                # pseudo_anomaly_shanghai_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_shanghai_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_shanghai_inpainting_smoothborder
                # if pseudo_anomaly_shanghai_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         shanghai_seq = next(shanghai_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         shanghai_iter = iter(shanghai_batch)
                #         shanghai_seq = next(shanghai_iter)
                #     net_in[b], mask = create_pseudoanomaly_seq_smoothborder(net_in[b], shanghai_seq[0], args.max_size,
                #                                                               args.h, args.w,
                #                                                               args.dataset_type, max_move=args.max_move)
                #     shanghai_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     shanghai_inpainting_smoothborder_pseudo_stat.append(False)


                # # imagenet inpainting smooth border pseudo anomaly
                # pseudo_anomaly_imagenet_inpainting_smoothborder = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_imagenet_inpainting_smoothborder
                # total_pseudo_prob += args.pseudo_anomaly_imagenet_inpainting_smoothborder
                # if pseudo_anomaly_imagenet_inpainting_smoothborder:
                #     try:
                #         # Samples the batch
                #         imagenet_img, _ = next(imagenet_iter)
                #     except StopIteration:
                #         # restart the generator if the previous generator is exhausted.
                #         imagenet_iter = iter(imagenet_batch)
                #         imagenet_img, _ = next(imagenet_iter)
                #     net_in[b], mask = create_pseudoanomaly_cifar_smoothborder(net_in[b], imagenet_img[0], args.max_size,
                #                                                               args.h, args.w,
                #                                                               args.dataset_type, max_move=args.max_move)
                #     imagenet_inpainting_smoothborder_pseudo_stat.append(True)
                #     pseudo_bool = True
                # else:
                #     imagenet_inpainting_smoothborder_pseudo_stat.append(False)


                # if pseudo_bool:
                #     cls_labels.append(0)
                # else:
                #     cls_labels.append(1)

            ########## TRAIN GENERATOR
            # net_in (batch_size,3,16,H,W)
            Recon_frames = model.forward(net_in)
            outputs = Recon_frames['output']
            att_w = Recon_frames['att']
            recon_index = Recon_frames['recon_index']
            # memory entropy loss
            entropy_loss = tr_entropy_loss_func(att_w)#weight entropy loss
            entropy_loss_val = entropy_loss.item()
            loss_entropy = entropy_loss_weight * entropy_loss
            cls_labels = torch.Tensor(cls_labels).unsqueeze(1).cuda()
            #recon loss
            loss_mse = loss_func_mse(outputs, net_in)
            #period loss
            loss_period = F.cross_entropy(recon_index,img_index)
            
            loss_period = loss_period * period_loss_weight

            modified_loss_mse = []
            for b in range(args.batch_size):
                # if jump_inpainting_pseudo_stat[b]:
                #     modified_loss_mse.append(torch.mean(loss_func_mse(outputs[b], imgsjump[1][b].to(outputs.device))))
                #     pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                #     pseudolosscounter += 1

                # else:  # no pseudo anomaly or cifar_inpainting_pseudo_stat[b] or cifar_inpainting_smooth_pseudo_stat[b] or cifar_inpainting_smoothborder_pseudo_stat[b] or ped2_inpainting_smoothborder_pseudo_stat[b] or shanghai_inpainting_smoothborder_pseudo_stat[b] or cifar_inpainting_cutmix_pseudo_stat[b] or cifar_inpainting_mixupcutmix_pseudo_stat[b]

                #     if cifar_inpainting_smooth_pseudo_stat[b] or cifar_inpainting_smoothborder_pseudo_stat[b] or imagenet_inpainting_smoothborder_pseudo_stat[b] or ped2_inpainting_smoothborder_pseudo_stat[b] or shanghai_inpainting_smoothborder_pseudo_stat[b] or cifar_inpainting_cutmix_pseudo_stat[b] or cifar_inpainting_mixupcutmix_pseudo_stat[b] or SW_video_inpainting_smoothborder_pseudo_stat[b] or VAD_inpainting_smoothborder_pseudo_stat[b]:
                #         new_loss_mse = loss_func_mse(outputs[b], imgs.cuda()[b])
                #         modified_loss_mse.append(torch.mean(new_loss_mse))
                #         pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                #         pseudolosscounter += 1
                #     else:
                #         modified_loss_mse.append(torch.mean(loss_mse[b]))
                #         lossepoch += modified_loss_mse[-1].cpu().detach().item()
                #         losscounter += 1

                modified_loss_mse.append(torch.mean(loss_mse[b]))
                lossepoch += modified_loss_mse[-1].cpu().detach().item()
                losscounter += 1

            assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss_recon = torch.mean(stacked_loss_mse)

            loss = loss_recon + loss_entropy + loss_period
            print('Loss: {:.6f}, Loss_recon: {:.6f}, Loss_entropy: {:.6f}, Loss_period: {:.6f}'.format(loss.item(),loss_recon.item(),loss_entropy.item(),loss_period.item()))
            # print('Loss: {:.6f}, Loss_recon: {:.6f}, Loss_entropy: {:.6f}'.format(loss.item(),loss_recon.item(),loss_entropy.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 10 == 0 or args.print_all:
                print("epoch {:d} iter {:d}/{:d}".format(epoch, j, len(train_batch)))
                print('Loss: {:.6f}'.format(loss.item()))

        print('----------------------------------------')
        print('Epoch:', epoch)
        if pseudolosscounter != 0:
            print('PseudoMeanLoss: Reconstruction {:.9f}'.format(pseudolossepoch/pseudolosscounter))
        if losscounter != 0:
            print('MeanLoss: Reconstruction {:.9f}'.format(lossepoch/losscounter))

        # Save the model and the memory items
        model_dict = {
            'model': model,
            'optimizer': optimizer.state_dict(),
        }
        if (epoch+1)%5 == 0:
            torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))

toc = time.time()
print('Training is finished')
print('Training time: ',(toc-tic))
sys.stdout = orig_stdout
f.close()



