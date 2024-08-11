import os

def display_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            os.system(f"img2txt -W {os.get_terminal_size().columns} -c center {image_path}")

# Replace "/path/to/skyfish/folder" with the actual path to the "skyfish" folder
display_images("skyfish/")