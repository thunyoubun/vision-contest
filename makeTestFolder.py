import os
import shutil

menus = ["Burger", "Dimsum","Ramen","Sushi"]

src_path = "C:/Users/thun_/Desktop/vision-contest/Dataset/test/"
des_path = "C:/Users/thun_/Desktop/vision-contest/TestImage"

with open("filelist.txt", "r") as file:
    filenames = [line.strip() for line in file]

image_paths = []
for menu in menus:
    folder_path = src_path + menu 
    for file_name in filenames:
        file_path = os.path.join(folder_path,file_name)   
        if os.path.isfile(file_path):   
            image_paths.append(file_path)

for file_name in image_paths:
    source_path = file_name
    shutil.copy(source_path, des_path)
 