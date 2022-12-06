import cv2
import os
import pathlib
import numpy as np
import glob
import imageio
from tqdm import tqdm
import os
import moviepy.video.io.ImageSequenceClip

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def video_to_frame(video_path, path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(path + '/frame_%04d.jpg' % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    


# def frame_to_video(image_path, fname, framerate=24):
    
#     # image_names = os.listdir(image_path)
#     # image_names = sorted(image_names)
#     # images = []
#     # for i in image_names:
#     #     images.append(image_path + "/" + i)
        
    
#     image_folder = image_path + '/*'
#     video_name = fname              #save as .avi
    
#     #is changeable but maintain same h&w over all  frames
#     img = cv2.imread(image_path + '/0.jpg')
#     height, width, layers = img.shape
    
#     #this fourcc best compatible for avi
#     fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     video=cv2.VideoWriter(video_name,fourcc, 60.0, (width,height))



#     for i in tqdm((sorted(glob.glob(image_folder),key=os.path.getmtime))):
#         x=cv2.imread(i)
#         video.write(x)

#     video.release()
    
#     # img_array = []
#     # for image in images:
#     #     img = cv2.imread(image)
#     #     height, width, layers = img.shape
#     #     size = (width, height)
#     #     img_array.append(img)

#     # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     # out = cv2.VideoWriter(fname,fourcc, framerate, size)
    
#     # for i in range(len(img_array)):
#     #     out.write(img_array[i])
#     # out.release()
    
def frame_to_video(image_path, fname, fps=1):

    # image_files = [os.path.join(image_folder,img)
    #             for img in os.listdir(image_folder)
    #             if img.endswith(".jpg")]
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    image_files = []
    for i in image_names:
        image_files.append(image_path + "/" + i)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(fname)
    
def image_to_GIF(image_path, fname):
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    image_files = []
    for i in image_names:
        image_files.append(imageio.imread(image_path + "/" + i))
    imageio.mimsave(fname, image_files)
    
    

path = '/home/mdsamiul/github_project/road_segmentation/data'
video_path = '/home/mdsamiul/github_project/road_segmentation/data/video.mp4'
# pathlib.Path(path+'/video_eval').mkdir(parents=True, exist_ok=True)
# video_to_frame(video_path, path+'/video_eval')
# frame_to_video('/home/mdsamiul/github_project/road_segmentation/prediction/fapnet/eval/best', '/home/mdsamiul/github_project/road_segmentation/prediction/fapnet/eval'+'/prediction.mp4')

# image_to_GIF('/home/mdsamiul/github_project/road_segmentation/prediction/fapnet/eval/best', '/home/mdsamiul/github_project/road_segmentation/prediction/fapnet/eval'+'/prediction.gif')
