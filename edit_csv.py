from PIL import Image
import glob
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import csv
import time
import seaborn as sns

import torch
import torch.nn.functional as F
# from scipy.linalg import svd
from pyts.decomposition import SingularSpectrumAnalysis

def write_to_csv(csv_path, data):
    with open(csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(data)

def record_rclass_from_csv_second_shortest(dir):
	csv_files = glob.glob(os.path.join(dir, '*.csv'))
	csv_files = [os.path.basename(file) for file in csv_files]
	
	STUDENT_NUM = len(csv_files)
	images_all = []
	emotions_all = []
	f_i_all = []

	for _ in range(STUDENT_NUM):
		images_all.append([])
		emotions_all.append([])
		f_i_all.append([])

	for student_id, filename in enumerate(csv_files):
		print(filename)
		with open(f"{dir}/{filename}", 'r') as file:
			csv_reader = csv.DictReader(file)
			for row in csv_reader:
				# Emotionの値を取得し、リストに追加
				images_all[student_id].append(row['Image'])
				emotions_all[student_id].append(row['Emotion'])
				f_i_all[student_id].append(float(row['F_i']))

		# w_iの記録用の新しいcsvファイルを作成
		header = ["Image", "Emotion", "F_i", "W_i"]
		output_csv_path = f'csv/output/rclass_{filename}'
		with open(output_csv_path, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(header)

	# リストの長さを取得して、リストを長さの昇順でソートする
	sorted_lists = sorted(images_all, key=len)
	# 2番目に短いリストの要素数を取得
	second_shortest_length = len(sorted_lists[1])
	shortest_length = len(sorted_lists[0])
	print("最も短いリストの要素数:", shortest_length)
	print("2番目に長いリストの要素数:", second_shortest_length)

	for frame in range(second_shortest_length):
		# 各生徒ごとのw_iを計算し、csvファイルに記録
		f_i_sum = 0
		flag_all_file = True
		shortest_length_index = -1
		for student_id, filename in enumerate(csv_files):
			if frame < shortest_length:
				# 最小フレーム数より小さい場合は全てのファイルを対象にw_iを計算
				f_i_sum += f_i_all[student_id][frame]
			else:
				flag_all_file = False
				# 最小フレームのファイルを除いたファイルを対象
				if frame < len(f_i_all[student_id]):
					f_i_sum += f_i_all[student_id][frame]
				else:
					shortest_length_index = student_id

		for student_id, filename in enumerate(csv_files):
			output_csv_path = f'csv/output/rclass_{filename}'
			# w_iを計算
			if f_i_sum == 0:
				w_i = 0
				# csvに記録
				write_to_csv(output_csv_path, [
					images_all[student_id][frame],
					emotions_all[student_id][frame],
					f_i_all[student_id][frame],
					w_i
				])
			else:
				if flag_all_file or student_id != shortest_length_index:
					w_i = f_i_all[student_id][frame] / f_i_sum
					# csvに記録
					write_to_csv(output_csv_path, [
						images_all[student_id][frame],
						emotions_all[student_id][frame],
						f_i_all[student_id][frame],
						w_i
					])

def record_rclass_from_csv_shortest(dir):
	csv_files = glob.glob(os.path.join(dir, '*.csv'))
	csv_files = [os.path.basename(file) for file in csv_files]
	
	STUDENT_NUM = len(csv_files)
	images_all = []
	emotions_all = []
	f_i_all = []

	for _ in range(STUDENT_NUM):
		images_all.append([])
		emotions_all.append([])
		f_i_all.append([])

	for student_id, filename in enumerate(csv_files):
		print(filename)
		with open(f"{dir}/{filename}", 'r') as file:
			csv_reader = csv.DictReader(file)
			for row in csv_reader:
				# Emotionの値を取得し、リストに追加
				images_all[student_id].append(row['Image'])
				emotions_all[student_id].append(row['Emotion'])
				f_i_all[student_id].append(float(row['F_i']))

		# w_iの記録用の新しいcsvファイルを作成
		header = ["Image", "Emotion", "F_i", "W_i"]
		output_csv_path = f'csv/output/rclass_{filename}'
		with open(output_csv_path, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(header)

	# リストの長さを取得して、リストを長さの昇順でソートする
	sorted_lists = sorted(images_all, key=len)
	shortest_length = len(sorted_lists[0])
	print("最も短いリストの要素数:", shortest_length)

	for frame in range(shortest_length):
		# 各生徒ごとのw_iを計算し、csvファイルに記録
		f_i_sum = 0
		for student_id, filename in enumerate(csv_files):
				f_i_sum += f_i_all[student_id][frame]

		for student_id, filename in enumerate(csv_files):
			output_csv_path = f'csv/output/rclass_{filename}'
			# w_iを計算
			if f_i_sum == 0:
				w_i = 0
			else:
				w_i = f_i_all[student_id][frame] / f_i_sum
			# csvに記録
			write_to_csv(output_csv_path, [
				images_all[student_id][frame],
				emotions_all[student_id][frame],
				f_i_all[student_id][frame],
				w_i
			])

def record_fi_from_csv(csv_file_path, filename, start_frame):
	# csvのEmotionの列を読み込む
	images = []
	emotions = []
	with open(csv_file_path, 'r') as file:
		csv_reader = csv.DictReader(file)
		current_row = 1
		for row in csv_reader:
			# 指定する行まで飛ばす
			if current_row < start_frame:
				current_row += 1
				continue
			images.append(row['Image'])
			emotions.append(row['Emotion'])
 
	# 新しいcsvファイルの作成
	output_csv_path = f'csv/output/f{start_frame}_{filename}'
	header = ["Image", "Emotion", "P_i", "N_i", "F_i"]
	with open(output_csv_path, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(header)

	p_i = 0
	n_i = 0
	for index in range(len(images)):
		emotion = emotions[index]
		if emotion == "Negative":
			n_i += 1
		elif emotion == "Positive":
			p_i += 1

		log = 2 * p_i + n_i
		if log <= 0:
			f_i = 0
		else:
			f_i = math.log(log)
		# 必要な値を新しいcsvに記録
		write_to_csv(output_csv_path, [images[index], emotion, p_i, n_i, f_i])

def plot_graph(data_y, type):
    x = range(len(data_y))
    y = data_y
    
    if type=='Positive':
        plt.plot(x, y, color='#1f77b4', label=type, alpha=0.5, linewidth=0.5) # 青
    elif type=='Negative':
        plt.plot(x, y, color='#d62728', label=type, alpha=0.5, linewidth=0.5) # 赤
    elif type=='Neutral':
        plt.plot(x, y, color='#2ca02c', label=type, alpha=0.5, linewidth=0.5) # 緑
    elif type=='Surprise':
        plt.plot(x, y, color='#ff7f0e', label=type, alpha=0.5, linewidth=0.5) # オレンジ
    else:
        plt.plot(x, y, color='#1f77b4', linewidth=0.5) # 青

def draw_graph_emotion_individual(csv_file_path, left, right, step_x):
    emotion_values = ["NaN", "Surprise", "Negative", "Neutral", "Positive"]	# y軸の順番を決めるためのダミーデータ
    with open(csv_file_path, 'r') as file:
        # csv.DictReaderオブジェクトを作成
        csv_reader = csv.DictReader(file)
        
        # 各行を処理
        for row in csv_reader:
            # Emotionの値を取得し、リストに追加
            emotion_values.append(row['Emotion'])

    # データの準備
    x = []				# x軸の値
    y = emotion_values	# y軸の値
    for i in range(len(emotion_values)):
        x.append(i-5)

    # 新しい図（Figure）を作成し、横幅を調整
    plt.figure(figsize=(15, 5))

    # グラフの描画
    plt.plot(x, y, alpha=0.5, linewidth=0.5)

    # グラフにタイトルとラベルを追加
    plt.title(csv_file_path)  # グラフのタイトル
    plt.xlabel('Frame')            # x軸のラベル
    plt.ylabel('Emotion')            # y軸のラベル
    # y軸のラベルを指定
    # plt.yticks(['positive', 'negative', 'neutral', 'NaN'])

    # x軸の最小値を設定	
    plt.xticks(range(left, right+step_x, step_x))

    # グラフを表示
    plt.show()

def draw_graph_emotion_logit_individual(csv_file_path, classes, min, max, step_x, step_y, left, right, difference, type):
    plt.figure(figsize=(15, 5))
	
    probabilities_all = []
    row_num = 0
    for class_name in classes:
        probabilities_all.append([])
        
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            row_num += 1
            logits = []
            for class_name in classes:
                logits.append(float(row[f'logit_{class_name}']))
            probabilities = F.softmax(torch.tensor(logits), dim=0)  # ロジットをソフトマックス関数で変換（0〜1の確率）
            probabilities = probabilities.tolist()
            for j in range(len(classes)):
                probabilities_all[j].append(probabilities[j] * 100)
    if difference:
        differences = []
        for i in range(int(row_num)):
            probability_1 = probabilities_all[0][i]
            probability_2 = probabilities_all[1][i]
            differences.append(probability_1 - probability_2)
        plot_graph(differences, "default")
        plt.axhspan(0, max, facecolor='#d62728', alpha=0.3)
        plt.axhspan(min, 0, facecolor='lightgreen', alpha=0.5)
    else:
        conv_probabilities_all = noise_removal(probabilities_all, classes, type)
        for i, class_name in enumerate(classes):
            plot_graph(conv_probabilities_all[i], class_name)

    plt.title(f"{csv_file_path}")  # グラフのタイトル
    plt.xlabel('Frame')            # x軸のラベル
    if difference:
        plt.ylabel('Difference[%]')            # y軸のラベル
    else:
        plt.ylabel('Probability[%]')            # y軸のラベル

    plt.xticks(range(left, right+step_x, step_x))   # x軸の最小値と最大値を設定
    plt.ylim(min, max)
    plt.yticks(np.arange(min, max+step_y, step=step_y)) # y軸の最小値と最大値を指定
    plt.grid(axis='y', color = "gray", linestyle="--")  # 目盛り線
    plt.legend(loc='upper left', fontsize='small', labelspacing=0.5, framealpha=0.5)    # 凡例

    # グラフを表示
    plt.show()

def noise_removal(data_list, classes, type):
    smoothed_data = []
    window_size = 30

    if type == "SMA" or type == "WMA" or type == "EMA":   # 移動平均
        smoothed_data = moving_average(data_list, window_size, type)
    elif type == "SSA":   # Singular Spectrum Analysis (SSA)
        smoothed_data = ssa(data_list, window_size)
    else:
        smoothed_data = data_list
    return smoothed_data

def moving_average(data_list, window_size, type):
    smoothed_data = []
    if type == "SMA":   # 単純移動平均
        for data in data_list:
            smoothed_data.append(np.convolve(data, np.ones(window_size)/window_size, mode='valid').tolist())
    elif type == "WMA":     # 加重移動平均
        weights = np.arange(1, window_size + 1)
        for data in data_list:
            wma = []
            for i in range(len(data) - window_size + 1):
                window = data[i:i + window_size]
                weighted_avg = np.dot(window, weights) / weights.sum()
                wma.append(weighted_avg)
            smoothed_data.append(wma)
    elif type == "EMA":     # 指数移動平均
        for data in data_list:
            ema = []
            alpha = 2 / (window_size + 1)
            ema.append(data[0])  # 初期値として最初のデータポイントを使用
            for i in range(1, len(data)):
                ema_value = alpha * data[i] + (1 - alpha) * ema[-1]
                ema.append(ema_value)
            smoothed_data.append(ema)
    return smoothed_data

def ssa(data_list, window_size):
    smoothed_data = []

    for data in data_list:
        data = np.array(data)  # リストをNumpy配列に変換
		# SSAの適用
        ssa = SingularSpectrumAnalysis(window_size=window_size)
        X_ssa = ssa.fit_transform(data.reshape(1, -1))

        # トレンドと周期成分の再構築
        reconstructed = X_ssa[0, 0] + X_ssa[0, 1]
		
        smoothed_data.append(reconstructed)
        # n = len(data)
        # if not isinstance(data, list):
        #     raise ValueError("Input data should be a list")
        # if not all(isinstance(x, (int, float)) for x in data):
        #     raise ValueError("All elements in data should be integers or floats")
        # if window_size > n:
        #     raise ValueError("Window size must be less than or equal to the length of the data")
        
        # # Step 1: Embedding
        # K = n - window_size + 1
        # trajectory_matrix = np.column_stack([data[i:i+window_size] for i in range(K)])
        
        # # Step 2: Singular Value Decomposition (SVD)
        # U, Sigma, VT = svd(trajectory_matrix)
        
        # # Step 3: Reconstruct the trajectory matrix with a reduced rank
        # rank = 1  # You can adjust the rank based on the singular values
        # d = np.diag(Sigma[:rank])
        # reconstructed_trajectory_matrix = U[:, :rank] @ d @ VT[:rank, :]
        
        # # Averaging over the anti-diagonals to reconstruct the time series
        # reconstructed_data = np.zeros(n)
        # counts = np.zeros(n)
        # for i in range(K):
        #     reconstructed_data[i:i+window_size] += reconstructed_trajectory_matrix[:, i]
        #     counts[i:i+window_size] += 1
        # reconstructed_data /= counts
        # smoothed_data.append(reconstructed_data)

    return smoothed_data

def draw_graph_from_csv(csv_file_path, filename, row_name, min, max, step, left, right):
	data = []
	with open(csv_file_path, 'r') as file:
		csv_reader = csv.DictReader(file)
		for row in csv_reader:
			data.append(float(row[row_name]))

	# データの準備
	x = []				# x軸の値
	y = data	# y軸の値
	for i in range(len(data)):
		x.append(i)

	# 新しい図（Figure）を作成し、横幅を調整
	plt.figure(figsize=(15, 5))

	# グラフの描画
	plt.plot(x, y)

	plt.title(filename)  # グラフのタイトル
	plt.xlabel('Frame')            # x軸のラベル
	plt.ylabel(row_name)            # y軸のラベル

	# plt.ylim(min, max)  # y軸の最小値と最大値を指定
	plt.ylim(min, max)
	plt.yticks(np.arange(min, max+0.1, step=step))

	plt.xlim(left=left)  # x軸の最小値を指定
	plt.xlim(right=right)

	plt.grid(axis='y', color = "gray", linestyle="--")

	# グラフを表示
	plt.show()

def draw_graph_from_csv_all(dir, csv_files, row_name, min, max, step):
	# 新しい図（Figure）を作成し、横幅を調整
	plt.figure(figsize=(15, 5))

	for index, filename in enumerate(csv_files):
		data = []
		with open(f"{dir}/{filename}", 'r') as file:
			csv_reader = csv.DictReader(file)
			for row in csv_reader:
				data.append(float(row[row_name]))

		# データの準備
		x = []		# x軸の値
		y = data	# y軸の値
		for i in range(len(data)):
			x.append(i)

		# グラフの描画
		plt.plot(x, y, label=filename)
		# plt.plot(x, y, label=f"s{index+1}")

	plt.title(f"All Result {row_name}")  # グラフのタイトル
	plt.xlabel('Frame')            # x軸のラベル
	plt.ylabel(row_name)            # y軸のラベル

	# plt.ylim(min, max)  # y軸の最小値と最大値を指定
	plt.ylim(min, max)
	plt.yticks(np.arange(min, max+0.1, step=step))
	# 目盛り線
	plt.grid(axis='y', color = "gray", linestyle="--")
	# 凡例
	# plt.legend()
	plt.legend(loc='upper left', fontsize='small', labelspacing=0.5, framealpha=0.5)
	# plt.legend(loc='upper right', labelspacing=0.5, framealpha=0.5)
	# loc='upper right'、loc='lower right'

	# グラフを表示
	plt.show()
     
def main():
    '''
    個別の感情変化のグラフを描写
    '''
    # dir = "csv/input"
    # csv_files = glob.glob(os.path.join(dir, '*.csv'))
    # csv_files = [os.path.basename(file) for file in csv_files]
    # for filename in csv_files:
    #     csv_file_path = f"{dir}/{filename}"
    #     left = 0
    #     right = 3600    # 練習の動画(test2)
    #     step_x = 200
    #     # right = 12000    # 本番の動画(test4)
    #     draw_graph_emotion_individual(csv_file_path, left, right, step_x)

    '''
    個別の各ラベルの確率のグラフを描写
    '''
    dir = "csv/input"
    csv_files = glob.glob(os.path.join(dir, '*.csv'))
    csv_files = [os.path.basename(file) for file in csv_files]
    classes = ['Positive', 'Surprise', 'Negative', 'Neutral']
    # classes = ['Negative', 'Neutral']
    min = 0
    max = 100
    step_x = 200
    step_y = 10
    left = 0
    right = 3500    # 練習の動画(test2)
    # right = 12000    # 本番の動画(test4)
    difference = False   # 確率の差で表示するか
    type = "WMA"
    if difference:
        min = -100
        max = 100
        step_y = 10
    for filename in csv_files:
        csv_file_path = f"{dir}/{filename}"
        draw_graph_emotion_logit_individual(csv_file_path, classes, min, max, step_x, step_y, left, right, difference, type)

    '''
    csvファイルから開始行を指定してf_iを計算
    '''
    # dir = "csv/input"
    # start_frame = 3600
    # csv_files = glob.glob(os.path.join(dir, '*.csv'))
    # csv_files = [os.path.basename(file) for file in csv_files]
    # for filename in csv_files:
    # 	print(filename)
    # 	csv_file_path = f"{dir}/{filename}"
    # 	record_fi_from_csv(csv_file_path, filename, start_frame)

    '''
    csvファイルからr_class(w_i)を計算（2番目に短いファイルに合わせる）
    '''
    # dir = "csv/input"
    # record_rclass_from_csv_second_shortest(dir)

    '''
    csvファイルからr_class(w_i)を計算（最も短いファイルに合わせる）
    '''
    # dir = "csv/input"
    # record_rclass_from_csv_shortest(dir)

    '''
    csvファイルからグラフを描写（1人ずつ）
    '''
    # dir = "csv/input"
    # csv_files = glob.glob(os.path.join(dir, '*.csv'))
    # csv_files = [os.path.basename(file) for file in csv_files]
    # for filename in csv_files:
    # 	# filename = "rclass_test4_morita.csv"
    # 	csv_file_path = f"{dir}/{filename}"
    # # row = "F_i"
    # # min = 0
    # # max = 9
    # # step = 1

    # 	row = "W_i"
    # 	min = 0
    # 	max = 9
    # 	step = 1
    # 	left = 20
    # 	right = 50
    # 	draw_graph_from_csv(csv_file_path, filename, row, min, max, step, left, right)

    '''
    csvファイルからグラフを描写（全員）
    '''
    # dir = "csv/input"
    # csv_files = glob.glob(os.path.join(dir, '*.csv'))
    # csv_files = [os.path.basename(file) for file in csv_files]
    # # row = "F_i"
    # # min = 0
    # # max = 9
    # # step = 1

    # row = "W_i"
    # min = 0
    # max = 0.5
    # step = 0.1
    # draw_graph_from_csv_all(dir, csv_files, row, min, max, step)

if __name__ == '__main__':
    main()