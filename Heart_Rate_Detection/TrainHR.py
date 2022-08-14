
#Training code for Spatiotemporal model
import torch.nn as nn
import glob
import os
import torch
import torch.nn.functional as F
import argparse
import os
import glob
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PhysNet import *
import PulseDataset
import PulseSampler


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,  # 0.0001 0.00007
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('-f')
loss_global = []
loss_ = []
loss_global_test = []


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (net_input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        net_input = net_input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, x_visual, x,y = model(net_input)
        rPPG = (output - torch.nanmean(output)) / torch.std(output)  # normalize
        BVP_label = (target - torch.nanmean(target)) / torch.std(target)  # normalize
        print(rPPG.size(), BVP_label.size())

        loss = criterion(rPPG, BVP_label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    with open('S_train_log1.csv', 'a') as log:
        log.write("{}, {}, {}, {}\n".format(losses.val, losses.avg, top1.val, top1.avg))
    loss_global.append(losses.avg)
    loss_.append(losses.val)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (net_input, target) in enumerate(val_loader):

        net_input = net_input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output, x_visual, x, y = model(net_input)
            rPPG = (output - torch.nanmean(output)) / torch.std(output)  # normalize
            BVP_label = (target - torch.nanmean(target)) / torch.std(target)  # normalize

            loss = criterion(rPPG, BVP_label)

        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    with open('test_log1.csv', 'a') as log:
        log.write("{}, {}\n".format(losses.avg, top1.avg))
    loss_global_test.append(losses.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='CropModel.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
 
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, every):
 
    lr = args.lr * (0.5 ** (epoch // every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    #train_seq is arbitrary.
    train_sequence_list = "train_seq.txt"   
    root_dir = "root_dir"
    seq_list = []
    end_indexes = []
    with open(root_dir+"/"+train_sequence_list, 'r') as seq_list_file:
        for line in seq_list_file:
            seq_list.append(line.rstrip('\n'))

    i = 0
    for s in seq_list:
        i+=1
        sequence_dir = os.path.join(root_dir, s)
        if sequence_dir[-2:len(sequence_dir)] == '_1':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[0:len(fr_list) // 2]
        elif sequence_dir[-2:len(sequence_dir)] == '_2':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[len(fr_list) // 2: len(fr_list)]
        else:
            if os.path.exists(sequence_dir + '/cropped/'):
                fr_list = glob.glob(sequence_dir + '/cropped/*.png')
            else:
                fr_list = glob.glob(sequence_dir + '/*.png')
 
        end_indexes.append(len(fr_list))

    end_indexes = [0, *end_indexes]
    print(end_indexes)

    test_sequence_list = "sequence.txt"
    root_dir = "/content/drive/MyDrive/Stress Detection/PURECROPPED"
    seq_list = []
    end_indexes_test = []
    with open(root_dir+"/"+test_sequence_list, 'r') as seq_list_file:
        for line in seq_list_file:
            seq_list.append(line.rstrip('\n'))
    i = 0
    for s in seq_list:
        i += 1
        sequence_dir = os.path.join(root_dir, s)
        if sequence_dir[-2:len(sequence_dir)] == '_1':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[0:len(fr_list) // 2]
        elif sequence_dir[-2:len(sequence_dir)] == '_2':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[len(fr_list) // 2: len(fr_list)]
        else:
            if os.path.exists(sequence_dir + '/cropped/'):
                fr_list = glob.glob(sequence_dir + '/cropped/*.png')
            else:
                fr_list = glob.glob(sequence_dir + '/*.png')
        end_indexes_test.append(len(fr_list))

    end_indexes_test = [0, *end_indexes_test]
    print(end_indexes_test)

    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("initialize model...")
    seq_len = 32
    model = PhysNet(seq_len)

    model = torch.nn.DataParallel(model)
    model.cuda()

 
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
 
        transforms.ToTensor(),
        normalize
    ])

    sampler = PulseSampler(end_indexes, seq_len, False)
    sampler_test = PulseSampler(end_indexes_test, seq_len, False)

    pulse = PulseDataset(train_sequence_list, root_dir, seq_len=seq_len,
                                          length=len(sampler), transform=trans)
    pulse_test = PulseDataset(test_sequence_list, root_dir, seq_len=seq_len,
                                               length=len(sampler_test), transform=transforms.Compose([
                                                                                                transforms.ToTensor(),
                                                                                                normalize]))
 
    train_loader = torch.utils.data.DataLoader(
        pulse,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        pulse_test,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,  sampler=sampler_test)


    criterion = NegPearson()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
    print('starting training...')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, 3)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 's_Drop_3d_32_{}.tar'.format(epoch)))

    plt.plot(loss_global)
    plt.title('average training loss')
    plt.show()

    plt.plot(loss_global_test)
    plt.title('testing loss')
    plt.show()

    plt.plot(loss_global_test)
    plt.plot(loss_global)
    plt.grid()
    plt.legend(['test loss', 'train loss'])
    plt.savefig('loss.jpg')