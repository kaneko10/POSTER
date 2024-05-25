import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from data_preprocessing.dataset_custom import CustomDataset
from data_preprocessing.dataset_sequece import SequenceDataset

from utils import *
from models.emotion_hyp import pyramid_trans_expr
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix

import time
import math
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()

def test(dataset, conversion, reclasses, conversion_rules, image_size, record):
    args = parse_args()
    # M1 MacではGPUが利用できないので、CUDA設定を削除
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    # data_transforms_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    data_transforms_test = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((224, 224)),  # デフォルトはこっち
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5], std=[0.5])  # グレースケールの場合、平均と標準偏差は1つの値のみ
    ])

    num_classes = 7
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "custom":
        print("dataset: ", dataset)
        datapath = f'./data/Custom/{dataset}'
        num_classes = 7
        test_dataset = CustomDataset(datapath, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=image_size, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "sequence":
        print("dataset: ", dataset)
        datapath = f'./data/Sequence/{dataset}'
        num_classes = 7
        test_dataset = SequenceDataset(datapath, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=image_size, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')


    print("Loading pretrained weights...", args.checkpoint)
    # checkpoint = torch.load(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # model = model.cuda()
    # モデルをCPUに移動
    model = model.to(torch.device('cpu'))

    pre_labels = []
    gt_labels = []
    # 間接推定用
    conv_pre_labels = []
    conv_gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        # 間接推定用
        conv_bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets, paths) in enumerate(test_loader):
            # outputs, features = model(imgs.cuda())
            # targets = targets.cuda()
            # 画像とターゲットをCPUに移動
            outputs, features = model(imgs.to(torch.device('cpu')))
            targets = targets.to(torch.device('cpu'))
            _, predicts = torch.max(outputs, 1)
            _, predicts = torch.max(outputs, 1)
            print(f"予測: {predicts}")
            print(f"正解: {targets}")
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()

            # ラベルを間接推定に変換
            if conversion:
                conv_predicts, conv_targets = label_list_conversion(predicts, targets, test_dataset.classes, reclasses, conversion_rules)
                conv_correct_or_not = torch.eq(conv_predicts, conv_targets)
                conv_bingo_cnt += conv_correct_or_not.sum().cpu()
                conv_pre_labels += conv_predicts.cpu().tolist()
                conv_gt_labels += conv_targets.cpu().tolist()

            for path in paths:
                file_name = os.path.basename(path)
                print(file_name)

            if record:
                if conversion:
                    record_pred_emotion(dataset, test_dataset.classes, predicts, conv_predicts, paths, reclasses, conversion_rules)

        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")
        print(pre_labels)
        cm = confusion_matrix(gt_labels, pre_labels)
        # print(cm)

        if conversion:
            conv_acc = conv_bingo_cnt.float() / float(test_size)
            conv_acc = np.around(conv_acc.numpy(), 4)
            print(f"Test accuracy: {conv_acc:.4f}.")
            print(conv_pre_labels)
            conv_cm = confusion_matrix(conv_gt_labels, conv_pre_labels)

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        cm = np.array(cm)
        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  #
            plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)

        if args.dataset == "affectnet":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]  #
            plot_confusion_matrix(cm, labels_name, 'AffectNet', acc)

        if args.dataset == "affectnet8class":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]  #
            # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
            # 7: Contempt,
            plot_confusion_matrix(cm, labels_name, 'AffectNet_8class', acc)

        if args.dataset == "custom":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  # RAF-DBの場合
            # labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]  #　AffectNetの場合、違うかも
            plot_confusion_matrix(cm, labels_name, 'custom', acc)

            if conversion:
                labels_name = ['NE', 'NEU', 'PO', 'SU']  #
                plot_confusion_matrix(conv_cm, labels_name, 'conv_custom', conv_acc)

def get_subfolders(folder_path):
    # フォルダ内のすべての要素を取得
    folder_contents = os.listdir(folder_path)
    subfolders = []

    # フォルダ内の要素をチェックしてサブフォルダを抽出
    for item in folder_contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
            
            output_path = os.path.join("final", item)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

    return subfolders

def is_image_filename(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 画像の拡張子リスト
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def write_to_csv(csv_path, data):
    with open(csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(data)

def is_black_image_by_filesize(image_path, threshold_kb=1):
    """
    与えられた画像が真っ黒の画像かどうかを判定する（ファイルサイズを利用）
    threshold_kbは真っ黒の画像と見なす最大ファイルサイズ（KB）
    """
    # 画像ファイルのサイズを取得（バイト単位）
    file_size = os.path.getsize(image_path)
    
    # バイト単位からKB単位に変換
    file_size_kb = file_size / 1024
    # print(file_size_kb)
    
    # ファイルサイズが閾値以下であれば真っ黒の画像と判定
    if file_size_kb <= threshold_kb:
        return True
    else:
        return False

def label_list_conversion(predicts, targets, classes, reclasses, conversion_rules):
    predicts = predicts.tolist()
    targets = targets.tolist()
    conv_predicts = []
    conv_targets = []
    for i in range(len(predicts)):
        predict_class_name = classes[predicts[i]]
        target_class_name = classes[targets[i]]
        for rule in conversion_rules:
            if rule[0] == predict_class_name:
                conv_class_name = rule[1]
                conv_class_index = reclasses.index(conv_class_name)
                conv_predicts.append(conv_class_index)
                break
        for rule in conversion_rules:
            if rule[0] == target_class_name:
                conv_class_name = rule[1]
                conv_class_index = reclasses.index(conv_class_name)
                conv_targets.append(conv_class_index)
                break
    
    return torch.tensor(conv_predicts), torch.tensor(conv_targets)

def label_conversion(predict, reclasses, conversion_rules):
    for rule in conversion_rules:
        if rule[0] == predict:
            conv_class_name = rule[1]
            conv_class_index = reclasses.index(conv_class_name)
            return conv_class_name, conv_class_index

def record_pred_emotion(dataset, classes, predicts, conv_predicts, paths, reclasses, conversion_rules):
    predicts = predicts.tolist()
    conv_predicts = conv_predicts.tolist()

    image_names = []
    for path in paths:
        file_name = os.path.basename(path)
        image_names.append(file_name)

    csv_file_path = f'csv/seq_{dataset}.csv'

    if not os.path.exists(csv_file_path):
        header = ["Image", "Emotion_ind", "Emotion", "P_i", "N_i", "F_i"]
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
        
    p_i = 0
    n_i = 0

    for i in range(len(image_names)):
		# # 黒画像か判断
        # image_path = f'./data/Custom/{dataset}/{image_name}'
        # is_black = is_black_image_by_filesize(image_path)
        # if is_black:
        #     predict_class_name = "NaN"
        # else:
        predict_class_name = classes[predicts[i]] 
        conversion_class_name, _ = label_conversion(predict_class_name, reclasses, conversion_rules)

        if conversion_class_name == "Negative":
            n_i += 1
        elif conversion_class_name == "Positive":
            p_i += 1

        log = 2 * p_i + n_i
        if log <= 0:
            f_i = 0
        else:
            f_i = math.log(log)

        write_to_csv(csv_file_path, [image_names[i], predict_class_name, conversion_class_name, p_i, n_i, f_i])
        
    print("CSVファイルに書き込みました。")

def main():
    '''
	指定したデータセットのテスト
	'''
    args = parse_args()
    dataset = "FER+"
    # image_size = 48
    image_size = 224    # デフォルト
    conversion = True       # ラベル変換を行うか
    record = False          # csvに記録するか
    if args.dataset == "sequence":
        record = True
    reclasses = ['Negative', 'Neutral', 'Positive', 'Surprise']
    conversion_rules = [
        ('Anger', 'Negative'), 
        ('Disgust', 'Negative'), 
        ('Fear', 'Negative'),
        ('Sadness', 'Negative'),
        ('Neutral', 'Neutral'),
        ('Happiness', 'Positive'),
        ('Surprise', 'Surprise')]
    test(dataset, conversion, reclasses, conversion_rules, image_size, record)

if __name__ == "__main__":                    
    main()

