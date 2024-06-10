import warnings

warnings.filterwarnings("ignore")
# from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os

import torch
import argparse
from data_preprocessing.dataset_finetuning import FinetuningDataset

from sklearn.metrics import f1_score, confusion_matrix
from time import time
from utils import *
from data_preprocessing.sam import SAM
from models.emotion_hyp import pyramid_trans_expr

import torch.optim as optim
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.00004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    return parser.parse_args()

def finetuning(dataset, image_size, epochs):
    args = parse_args()

    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize((224, 224)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "finetuning":
        print("dataset: ", dataset)
        datapath = f'./data/Finetuning/{dataset}'
        num_classes = 7
        train_dataset = FinetuningDataset(datapath, train=True, transform=data_transforms)
        val_dataset = FinetuningDataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=image_size, num_classes=num_classes, type=args.modeltype)

    val_num = val_dataset.__len__()
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    
    # 事前学習済みモデルのロード
    print("Loading pretrained weights...", args.checkpoint)
    # checkpoint = torch.load(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model = model.to(torch.device('cpu'))

    # パラメータの勾配を更新しないように設定
    set_parameter_requires_grad(model, feature_extracting=True)
    
    # モデル構造を確認し、最後の全結合層を新しいクラス数に合わせて変更
    # 例として、modelの最後の全結合層が`fc`である場合を仮定
    class_names = train_dataset.classes
    num_ftrs = model.head.linear.in_features
    model.head.linear = nn.Linear(num_ftrs, len(class_names))
    model = model.to(torch.device('cpu'))

    # 更新されるパラメータを確認
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    print(f"パラメータの数: {len(params_to_update)}")  # 更新されるパラメータの数

    # 損失関数と最適化手法の設定
    base_optimizer = torch.optim.Adam
    # base_optimizer = torch.optim.Adam(model.parameters(), lr=0.00004, weight_decay=1e-4)
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.00004, rho=0.05, adaptive=False,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    # ChatGPTが出力した設定
    # criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        for batch_i, (imgs, targets, paths) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(torch.device('cpu'))
            outputs, features = model(imgs)
            targets = targets.to(torch.device('cpu'))

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, features = model(imgs)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)

            loss = 2 * lsce_loss + CE_loss
            loss.backward() # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)


            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (epoch, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets, paths) in enumerate(val_loader):
                outputs, features = model(imgs.to(torch.device('cpu')))
                targets = targets.to(torch.device('cpu'))

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            epoch, val_acc, val_loss, f1, total_socre))

            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./checkpoint', "epoch" + str(epoch) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')

    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             model.train()
    #             dataloader = train_loader
    #         else:
    #             model.eval()
    #             dataloader = val_loader

    #         train_loss = 0.0
    #         correct_sum = 0
    #         iter_cnt = 0
    #         val_loss = 0.0
    #         bingo_cnt = 0

    #         for batch_i, (imgs, targets, paths) in enumerate(dataloader):
    #             iter_cnt += 1
    #             imgs = imgs.to(torch.device('cpu'))
    #             targets = targets.to(torch.device('cpu'))
    #             optimizer.zero_grad()

    #             with torch.set_grad_enabled(phase == 'train'):
    #                 outputs = model(imgs)
    #                 _, preds = torch.max(outputs, 1)
    #                 CE_loss = CE_criterion(outputs, targets)
    #                 loss = CE_loss
                    

    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()

    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(preds == labels.data)

    #         if phase == 'train':
    #             scheduler.step()

    #         epoch_loss = running_loss / dataset_sizes[phase]
    #         epoch_acc = running_corrects.double() / dataset_sizes[phase]

    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())

    #     print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # model.load_state_dict(best_model_wts)
    # return model

# パラメータの勾配を更新しないように設定
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():
    start_time = time()
    dataset = "Kaneko_v2"
    image_size = 224
    epochs = 200
    finetuning(dataset, image_size, epochs)
    end_time = time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    # # 事前学習済みモデルのロード
    # args = parse_args()
    # model = pyramid_trans_expr(img_size=image_size, num_classes=7, type=args.modeltype)
    # checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    # checkpoint = checkpoint["model_state_dict"]
    # model = load_pretrained_weights(model, checkpoint)
    # model = model.to(torch.device('cpu'))

    # # モデルの全ての属性を出力する
    # print(model)

if __name__ == "__main__":
    main()