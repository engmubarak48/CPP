#%%
import numpy as np
import pandas as pd


data = pd.read_json('/home/mohamud/CPP/36674_55880_bundle_archive/Indian_Number_plates.json', lines=True)
pd.set_option('display.max_colwidth', -1)

# Delete the empty column
del data['extras']

# Extract the points of the bounding boxes because thats what we want
data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

# And drop the rest of the annotation info
del data['annotation']

data.head()
#%%
#Download the data

Images = []
Plates = []

def downloadTraining(df):

    for index, row in df.iterrows():

        # Get the image from the URL
        resp = urllib.request.urlopen(row[0])
        im = np.array(Image.open(resp))

        # We append the image to the training input array
        Images.append(im)  

        # Points of rectangle
        x_point_top = row[1][0]['x']*im.shape[1]
        y_point_top = row[1][0]['y']*im.shape[0]
        x_point_bot = row[1][1]['x']*im.shape[1]
        y_point_bot = row[1][1]['y']*im.shape[0]

        # Cut the plate from the image and use it as output
        carImage = Image.fromarray(im)
        plateImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
        Plates.append(np.array(plateImage))


#%%
import cv2
image = os.listdir(os.getcwd() + '/coco128/images/train2017/')[0]
image_path = f'coco128/images/train2017/{image}'

image = cv2.imread(image_path)
print(image.shape)

cv2.imwrite(os.getcwd() + '/image_1.jpg', image)

#%%
name = '/gg/dd/ajaj.jama.ajvv.jpg'
osname = os.path.basename(name)
print(osname)
name = os.path.splitext(osname)[0]
print(name)

#%%
def convert_data_to_darknet_format(images_path, image_labels_text):

    image_labels = open(image_labels_text, 'r')
    image_labels = image_labels.readlines() 
    images = os.listdir(image_path)
    data_size = len(images)
    train_size = int(0.8 * data_size)
    test_size = int(0.2 * data_size) 

    for i, image_path, annotation in zip(range(data_size), images, image_labels):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[0:2]
        _, image_name, x, y, bbox_width, bbox_height = annotation.split()
        
        if image_name == os.path.basename(image_path):
            name = os.path.splitext(os.path.basename(image_path))[0]
            label_name = f"{name}.txt"
            x_center, y_center = x + bbox_width/2, y + bbox_height/2
            x_center, bbox_width = x_center/image_width, bbox_width/image_width
            y_center, bbox_height= y_center/image_height, bbox_height/image_height
            label_file = open(label_name, 'w')
            if i <= train_size:
                cv2.imwrite(f'{image_path}/train/{image_name}', image)
                label_file.write(f"{label_path}/{category_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")


#%%
def timeConversion(s):
    #
    # Write your code here.
    #

    if s[-2:] == 'AM' and s[:2] == '12':
        return '00' + s[2:-2]
    elif s[-2:] == 'AM':
        return s[:-2]
    elif s[-2:] == 'PM' and s[:-1] == '12':
        return s[:-2]
    else:
        return str(12 + int(s[:2])) + s[2:-2]

print(timeConversion('12:40:22AM'))

#%%
label_path, label_name = 'coco128', 'hhhhhhh.txt'
file = open(f'{label_path}/{label_name}', mode='w')
file.write('ahahha')