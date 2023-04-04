import re
import os
import glob
import shutil
from urllib.parse import urlencode, urljoin
import sys
import requests
import youtube_dl
from bs4 import BeautifulSoup
import pandas as pd
import unicodecsv as csv
from fuzzywuzzy import fuzz
import lxml
import cv2
from ultralytics import YOLO
import urllib.request
import random


def find_video_links(webpage_html):
    """ Extracts the portion of the link that points to
    the video file """
    expression = r'"(/sporty-v.*?)" target'
    return re.findall(expression, webpage_html)


def find_pitch_types(webpage_html):
    """ Extracts the portion of the tag that points
     to the pitch type """
    expression = r'search-pitch-label-.*?</span>'
    return re.findall(expression, webpage_html)


def download_video(url, name):
    """ Downloads the video file, according to the
     pitch url """
    url_2 = f"https://baseballsavant.mlb.com{url}"
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_2])


def download_all_matches(matches):
    """ Download all pitches video files """
    i = 1
    for match in matches:
        print(f"Downloading video {i} of {len(matches)} with url: https://baseballsavant.mlb.com{match}")
        download_video(match, i)
        i += 1


def rename(directory):
    """ Rename all the downloaded video files """
    os.chdir(directory)
    num = 1
    for file in [file for file in sorted(os.listdir(), key=os.path.getctime, reverse=False) if os.path.splitext(file)[1] == ".mp4"]:
        if os.path.splitext(file)[1] == ".mp4":
            os.rename(file, matching_df['MLBNAME'].iloc[0] + " - ".join(log_videos[1:]) + " - {:03d}.mp4".format(num))
            num += 1


def save_to_file(player, player_data):
    """ Save the full data for each pitch to a csv file """
    if not player_data:
        print("No player data to save to file.")
        return

    player_name = player.replace(".", "").strip().lower()
    filename = "_".join(player_name.split(" "))
    filepath = os.path.join(os.getcwd(), filename + "_" + "_".join(log_videos[1:]) + ".csv") 
    with open(filepath, "wb") as csv_file:
        rows = []
        writer = csv.writer(
            csv_file,
            delimiter=",",
            quotechar="\""
        )

        header = player_data[0].keys()
        rows.append(header)

        for data in player_data:
            row = [data[key] for key in header]
            rows.append(row)

        reversed_rows = rows[:0:-1]
        rows = rows[:1] + reversed_rows
        writer.writerows(rows)

filename = 'best.pt'

if not os.path.isfile(filename):
    print('We need to download the model, proceeding...')
    url = f'https://download1586.mediafire.com/73inz0f80mhgFIVaBia4uZ5bh7I8zqqnU2LGopoJF2zgEwueJoAlhCYHNwxP9qj4eO-6LFVJjoHl4bTQJ-NAcQTsiDQ/ag8fmc41h53o9z6/{filename}'
    urllib.request.urlretrieve(url, filename)
    print(f'Downloaded {filename} to current directory.')
else:
    print(f'{filename} already exists in current directory.')



# Parameters for youtube
ydl_opts = {}


# Expects date arguments in the format 2019-05-11
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
season = start_date[:4]


# player_id = input("Enter player ID: ")


url_id_map='https://drive.google.com/file/d/1KdSy7hWrrvpBbDVlR07yv5xxjZdfKK2F/view?usp=share_link'
url_id_map='https://drive.google.com/uc?id=' + url_id_map.split('/')[-2]
df_id_map = pd.read_csv(url_id_map)

exect = True
player_name = ""
while exect:
    player_name = input('Enter player name - NAME LASTNAME: ')

    # Create an empty dataframe to store matching rows
    matching_df = pd.DataFrame(columns=df_id_map.columns)

    # Loop through each row of the original dataframe
    for index, row in df_id_map.iterrows():
        # Use the fuzzywuzzy library to compare the name variable to the value in column B
        # If the match score is above the threshold of 70, add the row to the matching dataframe
        if fuzz.token_sort_ratio(player_name, row["MLBNAME"]) > 75:
            matching_df = matching_df.append(row)
    if (len(matching_df)) == 0:
        print("There were no matches")
    else:
        for a in range(0, len(matching_df)):
            print(str(a+1) + "-" + str(matching_df['MLBNAME'].iloc[a]) + 
            ", " + str(matching_df['BIRTHDATE'].iloc[a])  + 
            ", " + str(matching_df['POS'].iloc[a])) 
        selection = int(input("Enter the number for your selection: "))
        matching_df = matching_df.iloc[[selection-1]]
    
    print("Your selection: ")
    print(str(matching_df['MLBNAME'].iloc[0]) + 
            ", " + str(matching_df['BIRTHDATE'].iloc[0])  + 
            ", " + str(matching_df['POS'].iloc[0]))
    test = input("Do you want to change the player? (y/n)")
    if (test.lower() == "n"):
        exect = False

player_id = str(int(matching_df['MLBID'].iloc[0]))

  
is_last_pitch_str = ""

is_last_pitch = is_last_pitch_str.lower() == "true"
flag = "is...last...pitch|" if is_last_pitch else ""
player_type = "pitcher" if matching_df['POS'].iloc[0] == "P" else "batter"
lookup = "pitchers" if player_type == "pitcher" else "batters"

advanced = input("Do you want to tweak the advanced parameters? (y/n): ")

hand1 = ""
hand2 = ""
while (advanced.upper() == "Y" or advanced.upper() == "YES"):
    if (player_type == "pitcher"):
        hand1 = input("Against which batter's handeness? (L/R/ALL): ")
        hand2 = ""
    else:
        hand1 = ""
        hand2 = input("Against which pitcher's handeness? (L/R/ALL): ")
    hand1 = hand1.upper()
    hand2 = hand2.upper()
   
    if (hand1 != "L" and hand1 != "R"):
        hand1 = ""
    if (hand2 != "L" and hand2 != "R"):
        hand2 = ""
   

    advanced = "N"

        
        
url = f"https://baseballsavant.mlb.com/statcast_search?hfPTM=&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={season}%7C&hfSit=&player_type={player_type}&hfOuts=&hfOpponent=&pitcher_throws={hand2}&batter_stands={hand1}&hfSA=&game_date_gt={start_date}&game_date_lt={end_date}&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag={flag}&{lookup}_lookup%5B%5D={player_id}&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details&player_id={player_id}"
r = requests.get(url, allow_redirects=True, timeout=100)
html = r.content
print(url)

open('site.txt', 'wb').write(html)

with open('site.txt', encoding='utf-8') as f:
    lines = f.read()

matches = find_video_links(lines)
matches.reverse()

soup = BeautifulSoup(r.text, "html.parser")
# Obtain information from tag <table>
table1 = soup.find('table', id=f"ajaxTable_{player_id}")

# Obtain every title of columns with tag <th>
headers = []

if table1:
    for i in table1.find_all('th'):
        title = i.text
        headers.append(title)


    df = pd.read_html(lines, encoding='latin1', header=0)[0]
    del df[df.columns[-1]]
    df = df.loc[::-1].reset_index(drop=True)


if len(matches) <= 1:
    print("ERROR, 0 matches found in request")
    sys.exit()

pitches = ['EP', 'CU', 'CH', 'SI', 'SL', 'FF', 'FA', 'FC',
           'KC', 'FS', 'CS', 'PO', 'IN', 'SC']
pitch = input("Enter pitch  type: ")
pitch = pitch.upper()
if pitch.upper() in pitches:
    found = df.index[df['Pitch'] == pitch.upper()].tolist()
    df2 = df.loc[df['Pitch'] == pitch.upper()]
    result = [matches[i] for i in found]
else:
    pitch = "ALL"
    found = df.index.tolist()
    df2 = df
    result = [matches[i] for i in found]

log_videos = [matching_df['MLBNAME'].iloc[0], start_date, end_date,
             pitch, str(len(df2))]
print(f'There are {log_videos[-1]} videos in total.')
down = input("Do you want to download the pitches? (y/n): ")

if (down.lower() == 'y'):
    print(f'How many videos do you want to download? (0 for all)')
    num_elements = int(input())
    if num_elements > 0:
        while len(result) > num_elements:
            del result[random.randint(0, len(result)-1)]
    download_all_matches(result)
    rename(os.getcwd())

    src_folder = os.getcwd()
    dst_folder = os.getcwd() + "/vids/"

    try:
        os.mkdir(dst_folder)
    except OSError as error:
        print(error)

    # Search files with .mp4 extension in source directory
    pattern = "\*.mp4"
    files = glob.glob(src_folder + pattern)

    # move the files with mp4 extension
    for file in files:
        # extract file name form file path
        file_name = os.path.basename(file)
        shutil.move(file, dst_folder + file_name)
        print('Moved:', file)

url_feed = "https://baseballsavant.mlb.com/feed"
query_params = {
    "warehouse": "True",
    "hfPTM": "" if pitch.upper() == "ALL" else pitch + "|",
    "hfAB": "",
    "hfGT": "",
    "hfPR": "",
    "hfZ": "",
    "hfStadium": "",
    "hfBBL": "",
    "hfNewZones": "",
    "hfPull": "",
    "hfC": "",
    "hfSea": season + "|",
    "hfSit": "",
    "player_type": player_type,
    "hfOuts": "",
    "hfOpponent": "",
    "pitcher_throws": hand2,
    "batter_stands": hand1,
    "hfSA": "",
    "game_date_gt": start_date,
    "game_date_lt": end_date,
    "hfMo": "",
    "hfTeam": "",
    "home_road": "",
    "hfRO": "",
    "position": "",
    "hfInfield": "",
    "hfOutfield": "",
    "hfInn": "",
    "hfBBT": "",
    "hfFlag": "",
    lookup + "_lookup[]": player_id,
    "metric_1": "",
    "group_by": "name",
    "min_pitches": "",
    "min_results": "",
    "min_pas": "",
    "sort_col": "pitches",
    "player_event_sort": "api_p_release_speed",
    "sort_order": "desc",
    "type": "details",
    "player_id": player_id
}

url2 = urljoin(url_feed, "?" + urlencode(query_params))

print(url_feed)
print(url2)

with requests.get(url_feed, params=query_params, timeout=100) as r:
    r.raise_for_status()
    response = r.json()

save_to_file(matching_df['MLBNAME'].iloc[0], response)

input_folder = 'vids'
output_folder = 'done'

# set the area of interest for motion detection
center_x = 1280 // 2
top_y = (720 // 3) * 2
bottom_y = 720

# set the threshold value for motion detection
threshold_value = 50

# set the number of delay frames to adjust motion detection timing
delay_frames = 120

# get a list of all video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

# check if the folder output_folder exists
if not os.path.exists(output_folder):
    # if it doesn't exist, create the folder 'done'
    os.makedirs(output_folder)

# get the number of video files already in the output folder
done_files = [f for f in os.listdir(output_folder) if f.endswith('.mp4')]
last_file_number = int(done_files[-1][5:9]) if done_files else 0

# process each video file in the input folder
for video_file in video_files:
    # initialize motion detection variables
    motion_detected = False
    start_frame = 0

    # open the input video file
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    # create the output video file with a new filename based on the last file number
    out_file = f"video{str(last_file_number + 1).zfill(5)}.mp4"
    out_path = os.path.join(output_folder, out_file)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # process each frame in the input video file
    while True:
        # read the next frame from the input video
        ret, frame = cap.read()
        if not ret:
            break

        # extract the area of interest and apply thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[top_y:bottom_y, center_x:(2 * center_x)]
        _, roi_thresh = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)

        # detect motion in the area of interest
        if cv2.countNonZero(roi_thresh) > 0 and not motion_detected:
            motion_detected = True
            start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) + delay_frames

        # write frames to output video if motion is detected
        if motion_detected and cap.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame and \
                cap.get(cv2.CAP_PROP_POS_FRAMES) < (start_frame + fps * 2):
            video_out.write(frame)

    # release resources
    cap.release()
    video_out.release()

    # increment the last file number
    last_file_number += 1

    
   # set the directory path for the 'done' folder
directory = "done"

# get a list of all video files in the 'done' folder
video_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".mp4")]

# sort the list of video files alphabetically
video_files.sort()

# create a list of VideoCapture objects from the video files
video_captures = [cv2.VideoCapture(os.path.join(directory, f)) for f in video_files]

# get the first video dimensions and frame rate
width = int(video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_captures[0].get(cv2.CAP_PROP_FPS)

# create a VideoWriter object to save the final video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('joined.mp4', fourcc, fps, (width, height))

# concatenate the video frames into a single video
for cap in video_captures:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

# release the video captures and writer
for cap in video_captures:
    cap.release()
out.release()


directory = "done"

YOLO_MODEL_PATH = r"best.pt"

model = YOLO(YOLO_MODEL_PATH)

model.predict(source="joined.mp4", project="results", name="prediction", show=False, save_txt=True, save=True, conf=0.40)



# Define paths
current_folder = os.getcwd()
prediction_folder = os.path.join(current_folder, "results", "prediction")
target_folder = current_folder

# Define file names
original_name = "joined.mp4"
new_name = "predict_joined.mp4"

# Build full paths
original_path = os.path.join(prediction_folder, original_name)
new_path = os.path.join(target_folder, new_name)

# Rename and move the file
os.rename(original_path, new_path)
print(f"File moved to {new_path}")


def count_class_16_files(folder_path):
    class_16_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt') and file_name.startswith(short_file+'_'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if int(line_parts[0]) == 16:
                        class_16_count += 1
                        break
    return class_16_count


# define paths to input and output files
video_file = 'joined.mp4'
short_file = video_file[:-4]
labels_folder = os.getcwd() + '/results/prediction/labels'
output_file = 'tails-' + video_file

# set up video writer
cap = cv2.VideoCapture(video_file)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

# loop through frames of input video
positions = []
total_balls = count_class_16_files(labels_folder)
counter = 0
whole = 0
frames_list = []
gradient = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    whole = whole + 1
    # get corresponding txt file for current frame
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    label_file = os.path.join(labels_folder, f'{short_file}_{frame_number}.txt')
    # check if txt file exists and contains object with class 16
    if os.path.isfile(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                
                line_parts = line.strip().split()
                if int(line_parts[0]) == 16:
                    counter = counter + 1
                    # add baseball position to positions list
                    x, y, w, h = map(float, line_parts[1:])
                    center_x = int(x * frame_size[0])
                    center_y = int(y * frame_size[1])
                    positions.append((center_x, center_y))
                    print(counter/whole)
                    gradient.append(counter/whole)
                    #if counter>=int(total_balls*0.95):
                    #    del positions[:2]

    frames_list.append(counter/whole)
    if len(frames_list) >= 3: # check if there are at least 3 elements in list
        diff1 = frames_list[-2] - frames_list[-3] # calculate difference between last two elements and second-to-last two elements
        diff2 = frames_list[-1] - frames_list[-2] # calculate difference between last two elements and last two elements
        if diff1 > 0 and diff2 < 0: # check if difference goes from positive to negative
            #positions = [] # reset list if condition is met
            # calculate the midpoint of the list
            midpoint = int(len(positions)*0.6)

            # delete the first half of the list
            del positions[:int(midpoint*1.25)]
    # draw dots for baseball in current frame
    for pos in positions:
        cv2.circle(frame, pos, 3, (0, 255, 0), -1)

    # write frame to output video
    out.write(frame)

# release video capture and writer resources
cap.release()
out.release()
cv2.destroyAllWindows()
