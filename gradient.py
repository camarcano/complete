
# import the required libraries
import numpy as np
import os
import cv2

gradient = [0.05555555555555555, 0.10526315789473684, 0.15, 0.06153846153846154, 0.07575757575757576, 0.08823529411764706, 0.10144927536231885, 0.11428571428571428, 0.1267605633802817, 0.14084507042253522, 0.14864864864864866, 0.15789473684210525, 0.16883116883116883, 0.1794871794871795, 0.189873417721519, 0.2, 0.20987654320987653, 0.21951219512195122, 0.2289156626506024, 0.23809523809523808, 0.2441860465116279, 0.25, 0.25842696629213485, 0.26666666666666666, 0.27472527472527475, 0.14942528735632185, 0.15428571428571428, 0.15555555555555556, 0.16022099447513813, 0.16483516483516483, 0.16939890710382513, 0.17391304347826086, 0.16923076923076924, 0.17346938775510204, 0.17766497461928935, 0.18181818181818182, 0.18592964824120603, 0.19, 0.19402985074626866, 0.19801980198019803, 0.2, 0.20388349514563106, 0.20574162679425836, 0.20952380952380953, 0.2132701421800948]


# set the path to the video file and the labels folder
video_file = 'joined.mp4'
labels_folder = os.getcwd() + '/results/prediction/labels'

# use OpenCV to read the video file and get the number of frames

video = cv2.VideoCapture(video_file)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# create a list of zeros with the same length as the number of frames
exist = [0] * num_frames

# iterate over the files in the labels folder
for filename in os.listdir(labels_folder):
    # check if the file is a YOLO txt file
    if filename.endswith('.txt'):
        # extract the frame number from the filename
        frame_num = int(filename.split('.')[0].split('_')[-1])
        # read the contents of the file
        with open(os.path.join(labels_folder, filename), 'r') as f:
            contents = f.read()
        # check if the file contains class 16
        if '16' in contents.split():
            exist[frame_num] = 1

# print the resulting list
print(exist)


# define colors
yellow = (255, 255, 0)
red = (255, 0, 0)

# define helper function to convert value to rgb color
def get_color(value):
    if value is None:
        return yellow
    elif value == 0:
        return (0, 0, 0)
    else:
        color_list = [(i, yellow) for i in range(0, len(value), 5) if value[i:i+5] == [1]*5]
        if len(color_list) == 0:
            return yellow
        else:
            color_list.append((len(value), red))
            colors = [yellow] + [tuple(map(int, np.interp([i], [0, 5], [yellow[j], red[j]]))) for i in range(1, 6) for j in range(3)] + [red]
            rgb_colors = []
            start_index = 0
            for i in range(len(color_list)):
                end_index = color_list[i][0]
                color = colors[i % len(colors)]
                for j in range(start_index, end_index):
                    rgb_colors.append(color)
                start_index = end_index
            return rgb_colors

# example usage
#exist = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, None]
rgb_colors = get_color(exist)
print(rgb_colors)