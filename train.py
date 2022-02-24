from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from binary_ssd import build_binary_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import neptune
#from eval_coco import test_net
from icecream import ic
from collections import OrderedDict

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='results/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--exp_name', default='distillation_SSD300_binary_student_backbone_no_skip',
                    help='Name of the experiment folder in results/')
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/media/apple/Datasets/coco/',
                    help='Dataset root directory path')
parser.add_argument('--cdc', default=1, type=int,
                    help='coefficient for classification distillation')
parser.add_argument('--rdc', default=1, type=int,
                    help='coefficient for localization distillation')
parser.add_argument('--normalization', default=True, type=str2bool,
                    help='normalization in the distillation')
parser.add_argument('--change_teacher', default=True, type=str2bool,
                    help='change_teacher')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# if not os.path.exists(args.save_folder):
#    os.mkdir(args.save_folder)

# initilize neptune
print('INIT')
neptune.init('uzair789/Distillation')

PARAMS = {'dataset': args.dataset,
          'exp_name': args.exp_name,
          'batch_size': args.batch_size,
          'cdc': args.cdc,
          'rdc': args.rdc}

print("CREATE EXP")
exp = neptune.create_experiment(
    name=args.exp_name,
    params=PARAMS,
    tags=[
        'SSD300',
        'binary',
        'COCO',
        'Sierra'])

def load_teacher_student(student, cfg):
    """ Funtion to initilize the student with a pretrained checkpoint and also
    load the teacher"""
    teacher_folder = 'results/SSD300_fp_teacher'
    teacher_checkpoint_path = os.path.join(teacher_folder,
                                           'ssd300_COCO_final.pth')
    teacher_checkpoint = torch.load(teacher_checkpoint_path)
    teacher_ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    teacher_ssd_net.load_state_dict(teacher_checkpoint)
    print('Teacher loaded! ', teacher_checkpoint_path)

    # Init student
    student_folder = 'results/SSD300_binary_student'
    student_checkpoint_path = os.path.join(student_folder,
                                           'ssd300_COCO_final.pth')
    student_checkpoint = torch.load(student_checkpoint_path)
    student.load_state_dict(student_checkpoint)
    print('Student Init successful with ', student_checkpoint_path)
    return teacher_ssd_net, student

def nlm(teacher, student):
    """Distillation loss
    teacher = []
    student = []

    Returns:
        loss value (float)
    """
    #ic(teacher.shape)
    #ic(student.shape)

    reg_output = student[0]
    reg_output_teacher = teacher[0]
    class_output = student[1]
    class_output_teacher = teacher[1]

    c_loss_distill = 0
    reg_loss_distill = 0
    for i in range(args.batch_size):
        if args.normalization:
            class_teacher = class_output_teacher[i]/ torch.norm(class_output_teacher[i])
            reg_teacher = reg_output_teacher[i] / torch.norm(reg_output_teacher[i])
            class_student = class_output[i] / torch.norm(class_output[i])
            reg_student = reg_output[i] / torch.norm(reg_output[i])
        else:
            class_teacher = class_output_teacher[i]
            reg_teacher = reg_output_teacher[i]
            class_student = class_output[i]
            reg_student = reg_output[i]

        c_loss = torch.norm(class_teacher - class_student)
        r_loss = torch.norm(reg_teacher - reg_student)

        c_loss_distill += c_loss
        reg_loss_distill += r_loss

    class_loss_distill = args.cdc * (c_loss_distill/args.batch_size)
    reg_loss_distill = args.rdc * (reg_loss_distill/args.batch_size)

    return class_loss_distill, reg_loss_distill

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                image_set='train2017',
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    # eval data
    # testset = COCODetectionTesting(args.dataset_root, [('2017', 'val')], None)

    output_folder = os.path.join(args.save_folder, args.exp_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    #ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    ssd_net = build_binary_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    args.distillation = True
    # if distillation then load pretrained model
    if args.distillation:
        net_teacher, net = load_teacher_student(net, cfg)
        #net_teacher = torch.nn.DataParallel(net_teacher)
        #net_teacher = net_teacher.cuda()
        #net_teacher.eval()

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)

    elif args.distillation:
        # doing not loading the model here because the model needs to be loaded
        # before nn.DataParallel for distillation. This is a bad way of doing
        # it.
        pass

    else:
        #vgg_weights = torch.load(args.save_folder + args.basenet)
        vgg_weights = torch.load('weights/' + args.basenet)
        print('Loading base network...')
        print('keys in state dict')
        print(vgg_weights.keys())
        ssd_net.vgg.load_state_dict(vgg_weights, strict=False)




    if args.cuda:
        net = net.cuda()


    if not args.resume and not args.distillation:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    checkpoint_suffixes = [x for x in range(5000, 399000, 5000)] + ["final"]
    c = 0
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.change_teacher and iteration%5000 == 0:
            teacher_folder = 'results/SSD300_fp_teacher'
            teacher_checkpoint_path = os.path.join(teacher_folder,
                                               'ssd300_COCO_{}.pth'.format(checkpoint_suffixes[c]))
            teacher_checkpoint = torch.load(teacher_checkpoint_path)
            #new_dict = OrderedDict()
            #for key in teacher_checkpoint.keys():
            #    if 'module' not in key:
            #        new_key = 'module.'+ key
            #    else:
            #        new_key = key
            #    new_dict[key] = teacher_checkpoint[key]
            net_teacher.load_state_dict(teacher_checkpoint)
            net_teacher = torch.nn.DataParallel(net_teacher)
            net_teacher = net_teacher.cuda()
            net_teacher.eval()
            c +=1
            print("teacher checkpoint loaded for ", teacher_checkpoint_path)

        # if iteration < 4990:
        #    continue
        net.train()
        '''
        print('Logging 1')
        exp.log_metric('Current lr', float(optimizer.param_groups[0]['lr']))
        exp.log_metric('Current epoch', int(epoch))
        exp.log_metric('Current iteration', int(iteration))
        print('Logging 1 done')
        '''

        # print('Current lr', float(optimizer.param_groups[0]['lr']))
        # print('Current epoch', int(epoch))
        # print('Current iteration', int(iteration))

        if iteration != 0 and (iteration % epoch_size == 0):
            # update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
            #                'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            # images = Variable(images.cuda())
            # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            with torch.no_grad():
                images = Variable(images.float().cuda())
                targets = [Variable(ann.cuda()) for ann in targets]

        else:
            # images = Variable(images)
            # targets = [Variable(ann, volatile=True) for ann in targets]
            with torch.no_grad():
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)

        #teacher forward pass
        loss_cd = 0
        loss_ld = 0
        if args.distillation:
            with torch.no_grad():
                out_teacher = net_teacher(images)

            # distillation
            loss_cd, loss_ld =  nlm(out_teacher, out)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)


        loss = loss_l + loss_c + loss_cd + loss_ld
        loss.backward()
        optimizer.step()
        t1 = time.time()
        #loc_loss += loss_l.data[0]
        #conf_loss += loss_c.data[0]
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        #print('logging 2')
        exp.log_metric('loc loss', float(loss_l.item()))
        exp.log_metric('conf loss', float(loss_c.item()))
        exp.log_metric('total loss', float(loss.item()))
        exp.log_metric('Distill Classification loss', float(loss_cd))
        exp.log_metric('Distill Regression loss', float(loss_ld))
        # print('logging 2 done')
        # print('loc loss', loss_l.item())
        # print('conf loss', loss_c.item())
        # print('total loss', loss.item())
        # print('---')

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            print(
                'iter ' +
                repr(iteration) +
                ' || Loss: %.4f ||' %
                (loss.item()),
                end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:

            print('Saving state, iter:', iteration)
            checkpoint_path = os.path.join(
                output_folder,
                'ssd300_' +
                args.dataset +
                '_' +
                repr(iteration) +
                '.pth')
            # torch.save(ssd_net.state_dict(), os.path.join(output_folder, 'ssd300_'+ args.dataset + '_' +
            #           repr(iteration) + '.pth'))
            torch.save(ssd_net.state_dict(), checkpoint_path)

            # Do the eval every 5000 iterations
            #test_net(
            #    output_folder, checkpoint_path, args.cuda, testset, BaseTransformTesting(
            #        300, rgb_means=(
            #            123, 117, 104), rgb_std=(
            #            1, 1, 1), swap=(
            #            2, 0, 1)))

    torch.save(ssd_net.state_dict(), os.path.join(output_folder,
               'ssd300_' + args.dataset + '_final.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(X=torch.ones((1, 3)).cpu() * iteration,
             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() /
             epoch_size, win=window1, update=update_type)
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(e)
