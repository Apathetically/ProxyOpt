from model import U_Net
import numpy as np
import torch
from tools.dataloader import train_dataloader, test_dataloader, val_dataloader
from tqdm import tqdm
import visdom
from matplotlib import pyplot as plt
from PIL import Image
from tools.utils import get_psnr
import pickle
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

random.seed(1)

def train_step1():
    MAX_EPOCH = 5000
    LR = 0.003

    viz = visdom.Visdom(env='step1')
    image_dir = '/home/tongtong/project/raw_camera/ISPNetv3/SIDD_crop_bm3d'
    train_loader = train_dataloader(image_dir, batch_size=5, num_threads=16, img_size=512)
    test_loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    train_loader2 = train_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    net = U_Net(3, 3, step_flag=1, img_size=512)
    net = torch.nn.DataParallel(net).cuda()

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), LR)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75, 100, 125, 150, 175, 200], 0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)

    maxx = -1

    if not os.path.exists('./checkpoints_step1'):
        os.mkdir('./checkpoints_step1')

    for epoch in range(MAX_EPOCH):
        train_loss = 0.0
        net.train()
        for _, noisy, red, param in tqdm(train_loader):
            noisy = noisy.cuda()
            param = param.cuda()
            red = red.cuda()
            out = net(noisy, param)
            # compute loss.
            loss = criterion(out, red)
            train_loss += float(loss)
            # backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        psnr_gt_red_list = []
        psnr_gt_net_list = []

        psnr_gt_red_list_train = []
        psnr_gt_net_list_train = []

        # psnr for train dataset
        net.eval()
        for gt, noisy, red, param in tqdm(train_loader2):
            noisy_ = noisy.cuda()
            param_ = param.cuda()
            out = net(noisy_, param_)

            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list_train.append(float(psnr_gt_red))
            psnr_gt_net_list_train.append(float(psnr_gt_net))

        # psnr for test dataset
        net.eval()
        for gt, noisy, red, param in tqdm(test_loader):
            noisy_ = noisy.cuda()
            param_ = param.cuda()
            out = net(noisy_, param_)

            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list.append(float(psnr_gt_red))
            psnr_gt_net_list.append(float(psnr_gt_net))

        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(np.array(psnr_gt_net_list_train).mean())]), win='psnr_train', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(np.array(psnr_gt_net_list).mean())]), win='psnr_test', update='append' if epoch > 0 else None)
        print('epoch {:03d}, train red vs net : {:.3f} {:.3f}'.format(epoch, float(np.array(psnr_gt_red_list_train).mean()), float(np.array(psnr_gt_net_list_train).mean())))     
        print('epoch {:03d}, test  red vs net : {:.3f} {:.3f}'.format(epoch, float(np.array(psnr_gt_red_list).mean()), float(np.array(psnr_gt_net_list).mean())))

        save_path = './checkpoints_step1/net_iter{:03d}.pth'.format(epoch+1)
        torch.save(net.state_dict(), save_path)

        # if float(np.array(psnr_gt_net_list).mean()) > maxx:
        #     save_path = './checkpoints_step1_net_iter{:03d}.pth'.format(epoch+1)
        #     torch.save(net.state_dict(), save_path)
        #     maxx = float(np.array(psnr_gt_net_list).mean())

def train_step2():

    if not os.path.exists('./result'):
        os.mkdir('./result')

    MAX_EPOCH = 500
    LR = 0.002

    viz = visdom.Visdom(env='step2')
    image_dir = '/home/tongtong/project/raw_camera/ISPNetv3/SIDD_crop_bm3d'
    train_loader = train_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    test_loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    net = U_Net(3, 3, step_flag=3, img_size=512)

    net = torch.nn.DataParallel(net).cuda()
    # 237 for max_psnr and 459 for min loss
    net.load_state_dict(torch.load('./checkpoints_step1/net_iter237.pth'))

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam([net.module.param_layer], LR)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    maxx = -1

    if not os.path.exists('./checkpoints_step2'):
        os.mkdir('./checkpoints_step2')
    if not os.path.exists('./pickles_step2'):
        os.mkdir('./pickles_step2')

    params_list = [[] for i in range(5)]

    vis_size = 512

    param_layer_ = net.module.return_param_layer()
    '''
    data = []
    for idx in range(5):
        data.append(np.array(param_layer_[0, idx, :vis_size, :vis_size].detach().cpu())) 
    for idx in range(5):
        im = plt.imshow(data[idx], vmin=0, vmax=1, cmap='OrRd')
        plt.colorbar(im)
        plt.axis('off')
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        frame.axes.get_xaxis().set_visible(False)
        if not os.path.exists('./result/param{:1d}'.format(idx+1)):
            os.mkdir('./result/param{:1d}'.format(idx+1))
        plt.savefig('./result/param{:1d}/{:03d}.png'.format(idx+1, 0))
        plt.close()
        params_list[idx].append(imageio.imread('./result/param{:1d}/{:03d}.png'.format(idx+1, 0)))
    '''

    for epoch in range(MAX_EPOCH):
        train_loss = 0.0
        net.eval()
        for gt, noisy, _, __ in tqdm(train_loader):
            optimizer = torch.optim.Adam([net.module.param_layer], LR)
            noisy = noisy.cuda()
            gt = gt.cuda()
            out = net(noisy)
            # compute loss.
            loss = criterion(out, gt)
            # for idx in range(5):
            #     loss += 0.01*torch.var(net.module.param_layer[0, idx, :, :])
            train_loss += float(loss)
            # backward.
            optimizer.zero_grad()
            loss.backward()
            # print(net.module.param_layer.grad.data[0, 0, :vis_size, :vis_size].mean())
            for idx in range(5):
                avg_grad = net.module.param_layer.grad.data[0, idx, :, :].mean().float()
                net.module.param_layer.grad.data[0, idx, :, :] = avg_grad

            optimizer.step()
            net.module.update_param()

        psnr_gt_red_list = []
        psnr_gt_net_list = []

        net.eval()
        for gt, noisy, red, param in tqdm(test_loader):
            noisy_ = noisy.cuda()
            out = net(noisy_)

            psnr_gt_red = get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))
            psnr_gt_net = get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))

            psnr_gt_red_list.append(float(psnr_gt_red))
            psnr_gt_net_list.append(float(psnr_gt_net))

        res = net.module.return_param_value()
        print(res)

        if (epoch+1) % 2 == 0:
            LR *= 0.8
        if (epoch+1) % 30 == 0:
            LR = 0.002

        # generate gif
        param_layer_ = net.module.return_param_layer()
        '''
        data = []
        for idx in range(5):
            data.append(np.array(param_layer_[0, idx, :vis_size, :vis_size].detach().cpu())) 
        for idx in range(5):
            plt.imshow(data[idx], vmin=0, vmax=1, cmap='OrRd')
            plt.colorbar(im)
            plt.axis('off')
            frame = plt.gca()
            frame.axes.get_yaxis().set_visible(False)
            frame.axes.get_xaxis().set_visible(False)
        # plt.show()
            plt.savefig('./result/param{:1d}/{:03d}.png'.format(idx+1, epoch+1))
            plt.close()
            params_list[idx].append(imageio.imread('./result/param{:1d}/{:03d}.png'.format(idx+1, epoch+1)))
        '''
        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(np.array(psnr_gt_net_list).mean())]), win='loss1', update='append' if epoch > 0 else None)
        print('epoch {:03d}, red vs net : {:.3f} {:.3f}'.format(epoch, float(np.array(psnr_gt_red_list).mean()), float(np.array(psnr_gt_net_list).mean())))

        save_path = './checkpoints_step2/net_iter{:03d}.pth'.format(epoch+1)
        torch.save(net.state_dict(), save_path)
        param_layer_value = net.module.return_param_layer()

        viz.line(X=np.array([epoch]), Y=np.array([float(res[0])]), win='param1', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(res[1])]), win='param2', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(res[2])]), win='param3', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(res[3])]), win='param4', update='append' if epoch > 0 else None)
        viz.line(X=np.array([epoch]), Y=np.array([float(res[4])]), win='param5', update='append' if epoch > 0 else None)

        f = open('./pickles_step2/param_layer{:03d}.pkl'.format(epoch+1), 'wb')
        pickle.dump(param_layer_value, f)
        f.close()

        # if float(np.array(psnr_gt_net_list).mean()) > maxx:
        #     save_path = './checkpoints_step2/net_iter{:03d}.pth'.format(epoch+1)
        #     torch.save(net.state_dict(), save_path)
        #     param_layer_value = net.module.return_param_layer()
        #     f = open('./pickles_step2/param_layer{:03d}.pkl'.format(epoch+1), 'wb')
        #     pickle.dump(param_layer_value, f)
        #     f.close()
        #     maxx = float(np.array(psnr_gt_net_list).mean())
    '''
    for idx in range(5):
        imageio.mimsave('./result/param{:1d}/param{:1d}.gif'.format(idx+1, idx+1), params_list[idx], duration=1)
    '''

def test_step1():
    image_dir = '/home/tongtong/project/raw_camera/ISPNetv3/SIDD_crop_bm3d'
    loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    loader2 = train_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    net = U_Net(3, 3, step_flag=2, img_size=512)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('./checkpoints_step1/net_iter237.pth'))
    cnt = 0

    if not os.path.exists('./result'):
        os.mkdir('./result')
    
    # test dataset
    psnr_gt_red = []
    psnr_gt_net = []

    net.eval()
    for gt, noisy, red, param in loader:

        noisy_ = noisy.cuda()
        param_ = param.cuda()
        out = net(noisy_, param_)

        psnr_gt_red.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))))
        psnr_gt_net.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))))

        # plt
        gt = np.array(gt)[0].transpose(1, 2, 0)
        noisy = np.array(noisy)[0].transpose(1, 2, 0)
        red = np.array(red)[0].transpose(1, 2, 0)
        out = out.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(gt)
        plt.subplot(222)
        plt.imshow(noisy)
        plt.subplot(223)
        plt.imshow(red)
        plt.subplot(224)
        plt.imshow(out)

        if not os.path.exists('./result/figure_after_step1'):
            os.mkdir('./result/figure_after_step1')
        fig.savefig('./result/figure_after_step1/{}.png'.format(cnt))
        cnt += 1

    with open('./result/psnr_after_step1_test.txt', 'w') as f:
        for idx in range(len(psnr_gt_red)):
            f.write('psnr red vs net: {:.3f} {:.3f}\n'.format(psnr_gt_red[idx], psnr_gt_net[idx]))
        f.write('\navg  red vs net: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red).mean(), np.array(psnr_gt_net).mean()))
    f.close()

    # train_dataset
    psnr_gt_red_train = []
    psnr_gt_net_train = []

    net.eval()
    for gt, noisy, red, param in loader2:
        noisy_ = noisy.cuda()
        param_ = param.cuda()
        out = net(noisy_, param_)

        psnr_gt_red_train.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))))
        psnr_gt_net_train.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))))

    with open('./result/psnr_after_step1_train.txt', 'w') as f:
        for idx in range(len(psnr_gt_red_train)):
            f.write('psnr red vs net: {:.3f} {:.3f}\n'.format(psnr_gt_red_train[idx], psnr_gt_net_train[idx]))
        f.write('\navg  red vs net: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red_train).mean(), np.array(psnr_gt_net_train).mean()))


def test_step2():
    image_dir = '/home/tongtong/project/raw_camera/ISPNetv3/SIDD_crop_bm3d'
    loader = test_dataloader(image_dir, batch_size=1, num_threads=16, img_size=512)
    net = U_Net(3, 3, step_flag=4, img_size=512)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('./checkpoints_step2/net_iter191.pth'))
    f = open('./pickles_step2/param_layer{:03d}.pkl'.format(191), 'rb')
    param_layer = pickle.load(f)
    print(param_layer[0, 0, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 1, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 2, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 3, :, :].mean().cpu().detach().numpy(), end=' ')
    print(param_layer[0, 4, :, :].mean().cpu().detach().numpy())
    f.close()
    net.module.load_param_layer(param_layer)
    cnt = 0

    if not os.path.exists('./result'):
        os.mkdir('./result')

    psnr_gt_red = []
    psnr_gt_net = []

    net.eval()
    for gt, noisy, red, param in loader:

        noisy_ = noisy.cuda()
        param_ = param.cuda()
        out = net(noisy_)

        psnr_gt_red.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(red.cpu().detach()))))
        psnr_gt_net.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))))

        '''
        # plt
        gt = np.array(gt)[0].transpose(1, 2, 0)
        noisy = np.array(noisy)[0].transpose(1, 2, 0)
        red = np.array(red)[0].transpose(1, 2, 0)
        out = out.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(gt)
        plt.subplot(222)
        plt.imshow(noisy)
        plt.subplot(223)
        plt.imshow(red)
        plt.subplot(224)
        plt.imshow(out)

        if not os.path.exists('./result/figure_after_step2'):
            os.mkdir('./result/figure_after_step2')
        fig.savefig('./result/figure_after_step2/{}.png'.format(cnt))
        cnt += 1
        '''

    with open('./result/psnr_after_step2.txt', 'w') as f:
        for idx in range(len(psnr_gt_red)):
            f.write('psnr red vs net: {:.3f} {:.3f}\n'.format(psnr_gt_red[idx], psnr_gt_net[idx]))
        f.write('\navg red vs net: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red).mean(), np.array(psnr_gt_net).mean()))
    f.close()
    print('avg ori vs net : {:.3f} {:.3f}\n'.format(np.array(psnr_gt_red).mean(), np.array(psnr_gt_net).mean()))

def test_val():
    image_dir = '/home/tongtong/project/raw_camera/ISPNetv2/test_dataset'
    loader = val_dataloader(image_dir, num_threads=16, batch_size=1)
    net = U_Net(3, 3, step_flag=4)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('./checkpoints_step2/net_iter190.pth'))
    f = open('./pickles_step2/param_layer{:03d}.pkl'.format(190), 'rb')
    param_layer = pickle.load(f)
    f.close()
    net.module.load_param_layer(param_layer)
    cnt = 0

    if not os.path.exists('./result'):
        os.mkdir('./result')

    psnr_gt_noisy = []
    psnr_gt_net = []

    net.eval()
    for gt, noisy in loader:

        noisy_ = noisy.cuda()
        out = net(noisy_)

        psnr_gt_noisy.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(noisy.cpu().detach()))))
        psnr_gt_net.append(float(get_psnr(np.array(gt.cpu().detach()), np.array(out.cpu().detach()))))

        # plt
        gt = np.array(gt)[0].transpose(1, 2, 0)
        noisy = np.array(noisy)[0].transpose(1, 2, 0)
        out = out.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(gt)
        plt.subplot(132)
        plt.imshow(noisy)
        plt.subplot(133)
        plt.imshow(out)

        if not os.path.exists('./result/figure_after_val'):
            os.mkdir('./result/figure_after_val')
        fig.savefig('./result/figure_after_val/{}.png'.format(cnt))
        cnt += 1

    with open('./result/psnr_after_val.txt', 'w') as f:
        for idx in range(len(psnr_gt_net)):
            f.write('psnr ori vs net: {:.3f} {:.3f}\n'.format(psnr_gt_noisy[idx], psnr_gt_net[idx]))
        f.write('\navg ori vs net: {:.3f} {:.3f}\n'.format(np.array(psnr_gt_noisy).mean(), np.array(psnr_gt_net).mean()))
    f.close()

def get_param():
    f = open('./pickles_step2/param_layer{:03d}.pkl'.format(110), 'rb')
    param_layer = pickle.load(f).squeeze()
    f.close()
    print(param_layer[0, :, :].mean())
    print(param_layer[1, :, :].mean())
    print(param_layer[2, :, :].mean())
    print(param_layer[3, :, :].mean())
    print(param_layer[4, :, :].mean())

if __name__ == '__main__':
    train_step1()
    # test_step1()
    # train_step2()
    # test_step2()
    # test_val()
    # get_param()
