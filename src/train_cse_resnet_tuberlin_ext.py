import argparse
import os
import shutil
import time
import numpy as np
from ResnetModel import CSEResnetModel_KDHashing
from TUBerlin import TUBerlinDataset
from Sketchy import SketchImagePairedDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import math
import random
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pretrainedmodels
from senet import cse_resnet50
import torch.nn.functional as F

from tool import AverageMeter, validate_paired, SupConLoss
from data.auto_augment import rand_augment_transform
from data_processing.RandAugment import RandAugment
from model import SemanticBatchHardXBM




img_size_min = 224
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
mean = IMAGENET_DEFAULT_MEAN
auto_augment = 'rand-m9-mstd0.5'
aa_params = dict(translate_const=int(img_size_min * 0.45),
               img_mean=tuple([min(255, round(255 * x)) for x in mean]),
               )

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for Sketchy Training')

parser.add_argument('--savedir', '-s', metavar='DIR',
                    default='../cse_resnet50/checkpoint/',
                    help='path to save dir')
parser.add_argument('--wv_size', type=int, default=300)

#======================== Architecture ===========================
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: cse_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=220,
                    help='number of classes (default: 100)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                    help='number of hashing dimension (default: 64)')

#======================== Training ==============================
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=100, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
                    help='freeze features of the base network')
parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                    help='use ems loss for the training')
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume from the latest epoch')

#======================== DataSet ==============================
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str,
                    help='zeroshot version for training and testing (default: zeroshot)')

#======================== SAKE ===========================
parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for kd loss (default: 1)')
parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                    help='lambda for semantic adjustment (default: 0.3)')
parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for total SAKE loss (default: 1)')
parser.add_argument('--cls_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for cross entropy loss (default: 1)')


#======================== Contrastive Learning ===========================
parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default=0.1, type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--temperature', metavar='LAMBDA', default=0.07, type=float,
                    help='lambda for temperature in contrastive learning')
parser.add_argument('--margin', metavar='MARGIN', default=0.2, type=float,
                        help='margin for contrastive loss')
parser.add_argument('--use_strong_aug', metavar='FLAG', default=1, type=int,
                        help='whether to use strong augmentation')


#======================== Cross-batch Semantic Metric Learning ===========================
parser.add_argument('--K', metavar='K', default=256, type=int,
                    help='size of memory bank')
parser.add_argument('--csm_lambda', metavar='LAMBDA', default=0.1, type=float,
                    help='lambda for Cross-batch Semantic Metric loss')
parser.add_argument('--xbm_start_iter', default=10000, type=int,
                    help='start iteration for xbm')


gpu_ids = [0,1]
gpu_ids = list(map(lambda x:str(x), gpu_ids))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)



def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

SEED=0
seed_torch(SEED)


class EMSLoss(nn.Module):
    def __init__(self, m=4):
        super(EMSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.m = m

    def forward(self, inputs, targets):
        mmatrix = torch.ones_like(inputs)
        for ii in range(inputs.size()[0]):
            mmatrix[ii, int(targets[ii])] = self.m

        inputs_m = torch.mul(inputs, mmatrix)
        return self.criterion(inputs_m, targets)


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask=None, mask_pos=None):
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)

        if mask_pos is not None:
            target_logits = target_logits + mask_pos

        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(target_logits, dim=1))) / sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)), mask)) / sample_num

        return loss


def main():
    global args, iteration
    iteration = 0
    args = parser.parse_args()
    print(args)



    # create model
    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes,
                                     freeze_features=args.freeze_features, ems=args.ems_loss, module='CSE')

    # model.cuda()
    model = nn.DataParallel(model).cuda()

    print(str(datetime.datetime.now()) + ' student model inited.')
    model_t = cse_resnet50()
    model_t = nn.DataParallel(model_t).cuda()
    print(str(datetime.datetime.now()) + ' teacher model inited.')

    sbh_model = SemanticBatchHardXBM(args).cuda()

    # define loss function (criterion) and optimizer
    if args.ems_loss:
        print("**************  Use EMS Loss!")
        curr_m = 1
        criterion_train = EMSLoss(curr_m).cuda()
    else:
        # criterion_train = nn.CrossEntropyLoss().cuda()
        from tool import LabelSmoothingCrossEntropy
        criterion_train = LabelSmoothingCrossEntropy().cuda()

    criterion_contrastive = SupConLoss(args.temperature, m=args.margin).cuda()
    criterion_train_kd = SoftCrossEntropy().cuda()
    criterion_test = nn.CrossEntropyLoss().cuda()

    param_list = [
        {'params': model.parameters()},
        {'params': sbh_model.parameters()}
    ]
    optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])

    weak_contrastive_transform = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize([224, 224]),
                                                     # transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.RandomApply(
                                                         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                                         p=0.8),
                                                     transforms.RandomGrayscale(p=0.2),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(immean, imstd)])
    strong_contrastive_transform = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize([224, 224]),
                                                       # transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
                                                       transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.RandomApply(
                                                           [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                                           p=0.8),
                                                       transforms.RandomGrayscale(p=0.2),
                                                       # RandAugment(n=2, m=10),
                                                       rand_augment_transform(auto_augment, aa_params),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(immean, imstd),
                                                       ])
    if args.use_strong_aug:
        contrastive_transform = [weak_contrastive_transform, strong_contrastive_transform]
    else:
        contrastive_transform = [weak_contrastive_transform, weak_contrastive_transform]

    sketch_batch_size = args.batch_size // 10
    photo_batch_size = args.batch_size - sketch_batch_size

    tuberlin_train = TUBerlinDataset(split='train', zero_version=args.zero_version,
                                     transform=transformations, contrastive_transform=contrastive_transform,
                                     aug=True, cid_mask=True)
    train_loader = DataLoader(dataset=tuberlin_train,
                              batch_size=sketch_batch_size,
                              shuffle=True, num_workers=3,
                              drop_last=True)

    tuberlin_train_ext = TUBerlinDataset(split='train', zero_version=args.zero_version,
                                         version='ImageResized_ready',
                                         transform=transformations,
                                         contrastive_transform=contrastive_transform,
                                         aug=True, cid_mask=True)

    train_loader_ext = DataLoader(dataset=tuberlin_train_ext,
                                  batch_size=photo_batch_size, shuffle=True,
                                  num_workers=3)

    tuberlin_val = SketchImagePairedDataset(dataset='tuberlin',
                                           zero_version=args.zero_version,
                                           transform=transformations)
    val_loader = DataLoader(dataset=tuberlin_val, batch_size=args.batch_size // 2,
                            shuffle=True, num_workers=3,
                            drop_last=True)

    print(str(datetime.datetime.now()) + ' data loaded.')

    if args.evaluate:
        acc1 = validate(val_loader, model, criterion_test, criterion_train_kd, model_t)
        print('Acc is {.3f}'.format(acc1))
        return

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    savedir = f'tuberlin_kd({args.kd_lambda})_kdneg({args.kdneg_lambda})_sake({args.sake_lambda})_' \
              f'dim({args.num_hashing})_' \
              f'contrastive({args.contrastive_lambda}-{args.margin}-{args.use_strong_aug})_T({args.temperature})' \
              f'csm({args.csm_lambda}-{args.K})'

    if not os.path.exists(os.path.join(args.savedir, savedir)):
        os.makedirs(os.path.join(args.savedir, savedir))

    best_acc1 = 0
    start_epoch = 0

    if args.resume:
        # resume from a checkpoint
        resume = os.path.join(args.savedir, savedir, 'checkpoint.pth.tar')

        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']

            save_dict = checkpoint['state_dict']
            save_optimizer = checkpoint['optimizer']
            best_acc1 = checkpoint['best_acc1']
            model_dict = model.state_dict()

            trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
            print('trashed vars from resume dict:')
            print(trash_vars)

            resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
            # resume_dict['module.linear.cpars'] = save_dict['module.linear.weight']

            model_dict.update(resume_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(save_optimizer)

            # model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {} acc {:.3f})"
                  .format(resume, checkpoint['epoch'], checkpoint['best_acc1'].item()))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return

    val_accs = []
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        if args.ems_loss:
            if epoch in [20, 25]:
                new_m = curr_m * 2
                print("update m at epoch {}: from {} to {}".format(epoch, curr_m, new_m))
                criterion_train = EMSLoss(new_m).cuda()
                curr_m = new_m

        train(train_loader, train_loader_ext, model, criterion_train, criterion_train_kd, criterion_contrastive, \
              optimizer, epoch, model_t, sbh_model)
        if epoch > 20:
            acc1 = validate_paired(val_loader, model, criterion_test, criterion_train_kd, model_t)
            val_accs.append(acc1.item())
            print('Validated accuracy: {}'.format(val_accs))

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.savedir, savedir, 'checkpoint.pth.tar'))


def train(train_loader, train_loader_ext, model, criterion, criterion_kd, criterion_contrastive, \
          optimizer, epoch, model_t, sbh_model):
    global iteration
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_kd = AverageMeter()
    losses_contrastive = AverageMeter()
    losses_sbh = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    model_t.eval()
    end = time.time()
    for i, ((sketch, sketch1, sketch2, label_s, sketch_cid_mask, wv_s),
            (image, image1, image2, label_i, image_cid_mask, wv_i)) \
            in enumerate(zip(train_loader, train_loader_ext)):
        sketch, sketch1, sketch2, label_s, sketch_cid_mask, wv_s = sketch.cuda(), sketch1.cuda(), \
                                                                   sketch2.cuda(), torch.cat([label_s]).cuda(), \
                                                                   torch.cat([sketch_cid_mask]).cuda(), wv_s.cuda()
        image, image1, image2, label_i, image_cid_mask, wv_i = image.cuda(), image1.cuda(), \
                                                               image2.cuda(), torch.cat([label_i]).cuda(), \
                                                               torch.cat([image_cid_mask]).cuda(), wv_i.cuda()
        tag_zeros = torch.zeros(sketch.size()[0], 1)
        tag_ones = torch.ones(image.size()[0], 1)
        tag_all = torch.cat([tag_zeros, tag_ones], dim=0).cuda()

        sketch_shuffle_idx = torch.randperm(sketch.size(0))
        image_shuffle_idx = torch.randperm(image.size(0))
        sketch = sketch[sketch_shuffle_idx]
        sketch1 = sketch1[sketch_shuffle_idx]
        sketch2 = sketch2[sketch_shuffle_idx]
        wv_s = wv_s[sketch_shuffle_idx]
        label_s = label_s[sketch_shuffle_idx].type(torch.LongTensor).view(-1, ).cuda()
        sketch_cid_mask = sketch_cid_mask[sketch_shuffle_idx].float()

        image = image[image_shuffle_idx]
        image1 = image1[image_shuffle_idx]
        image2 = image2[image_shuffle_idx]
        wv_i = wv_i[image_shuffle_idx]
        label_i = label_i[image_shuffle_idx].type(torch.LongTensor).view(-1, ).cuda()
        image_cid_mask = image_cid_mask[image_shuffle_idx].float()

        target_all = torch.cat([label_s, label_i]).cuda()
        cid_mask_all = torch.cat([sketch_cid_mask, image_cid_mask]).cuda()

        output, output_kd, hash_code, features = model(torch.cat([sketch, image, sketch1, image1, sketch2, image2], 0),
                                                       torch.cat([tag_all, tag_all, tag_all], 0))
        output = output[:tag_all.size(0)]
        output_kd = output_kd[:tag_all.size(0)]


        contrastive_feature = F.normalize(hash_code[tag_all.size(0):])

        contrastive_feature1 = torch.unsqueeze(contrastive_feature[:tag_all.size(0)], 1)
        contrastive_feature2 = torch.unsqueeze(contrastive_feature[tag_all.size(0):], 1)

        with torch.no_grad():
            output_t = model_t(torch.cat([sketch, image], 0), tag_all)

        loss_sbh, x_sem = sbh_model(hash_code[:tag_all.size(0)],
                                    torch.cat([wv_s, wv_i]),
                                    torch.cat([label_s, label_i]),
                                    iter=iteration)
        sft_x_sem = model.module.linear(x_sem)
        entropy = torch.sum(-F.softmax(sft_x_sem, dim=1) * F.log_softmax(sft_x_sem, dim=1), dim=1)
        entropy_weight = (1 + torch.exp(-entropy))
        loss_sbh = (loss_sbh * entropy_weight).mean()

        #loss_sbh = torch.tensor(0.0)

        loss_cls = criterion(output, target_all)
        loss_contrastive = criterion_contrastive(torch.cat([contrastive_feature1, contrastive_feature2], 1),
                                                 target_all)
        loss_kd = criterion_kd(output_kd, output_t * args.kd_lambda, tag_all, cid_mask_all * args.kdneg_lambda)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
        losses_cls.update(loss_cls.item(), tag_all.size(0))
        losses_kd.update(loss_kd.item(), tag_all.size(0))
        losses_contrastive.update(loss_contrastive.item(), tag_all.size(0) * 2)
        losses_sbh.update(loss_sbh.item(), tag_all.size(0))
        top1.update(acc1[0], tag_all.size(0))
        top5.update(acc5[0], tag_all.size(0))

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss_total = args.cls_lambda * loss_cls + args.sake_lambda * loss_kd + \
                     args.contrastive_lambda * loss_contrastive + args.csm_lambda * loss_sbh

        loss_total.backward()
        optimizer.step()
        iteration += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        nb_batch = min(len(train_loader), len(train_loader_ext))

        if i % args.print_freq == 0 or i == nb_batch - 1:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                  'Loss_CLS {loss_cls.val:.3f} ({loss_cls.avg:.3f}) '
                  'Loss_KD {loss_kd.val:.3f} ({loss_kd.avg:.3f}) '
                  'Loss_Contra {loss_contrastive.val:.3f} ({loss_contrastive.avg:.3f}) '
                  'Loss_SBH {loss_sbh.val:.3f} ({loss_sbh.avg:.3f}) '
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})%'.format(
                epoch, i, nb_batch, batch_time=batch_time,
                loss_cls=losses_cls, loss_kd=losses_kd,
                loss_contrastive=losses_contrastive,
                loss_sbh=losses_sbh,
                top1=top1))


def validate(val_loader, model, criterion, criterion_kd, model_t):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        target = target.type(torch.LongTensor).view(-1, )
        target = torch.autograd.Variable(target).cuda()

        # compute output
        with torch.no_grad():
            output_t = model_t(input, torch.zeros(input.size()[0], 1).cuda())
            output, output_kd, _, __ = model(input, torch.zeros(input.size()[0], 1).cuda())

        loss = criterion(output, target)
        loss_kd = criterion_kd(output_kd, output_t)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 or i == len(val_loader) - 1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} ({loss.avg:.3f} {loss_kd.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, loss_kd=losses_kd,
                top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        filepath = '/'.join(filename.split('/')[0:-1])
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    # lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
    # epoch_curr = min(epoch, 20)
    # lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )
    lr = args.lr * math.pow(0.001, float(epoch) / args.epochs)
    print('epoch: {}, lr: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
