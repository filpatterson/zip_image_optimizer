import argparse
import zipfile
import tempfile
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#   loading LBP Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("data/lbpcascade_frontalface.xml")


def extract_zip_to_temp(archive_name: str, temp_directory: str="temp"):
    """extract all images out of the zip archive into temporary directory.
    IMPORTANT: consider that by default data is saved into 'temp' directory.

    Args:
        archive_name (str): name of archive to decompress
        temp_directory (str): name of directory where to save images
    """
    with zipfile.ZipFile(archive_name, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith((".png", ".jpg", ".jpeg", ".gif")):
                zip_ref.extract(file, path=temp_directory)
    zip_ref.close()
    

def get_image_filepaths(target_directory: str) -> list:
    """get paths to all images in the directory

    Args:
        target_directory (str): directory where images are stored

    Returns:
        list: paths to images
    """
    image_filepaths = []
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg", ".gif")):
                image_filepaths.append(os.path.join(root, file))
    return image_filepaths


def remove_temp_directory(temporary_directory: str):
    """remove temporary directory used to processing calculations

    Args:
        temporary_directory (str): name of directory to remove
    """
    for root, dirs, files in os.walk(temporary_directory, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temporary_directory)
    
    
def optimize_images_zip_archive(args):
    """Take zip archive from given filename or path in 'target_archive_path',
    extract images out of it into temporary directory that will be created with
    'temp_dir_name' name, then from all images will be taken face and transformed
    into JPEG format. Those optimized images are saved into 'compressed_temp_dir_name'
    directory. At the final stage, will be created archive with the name of original
    one with added '_compressed' flag, after which all temporary folders and 
    original zip are removed 

    Args:
        target_archive_path (str): name or path of archive to optimize
        temp_dir_name (str): name of directory where images will be extracted from
                original archive and that will be removed in the end of process
        compressed_temp_dir_name (str): name of directory where compressed images
                will be saved and that will be removed in the end of process.
    """
    target_archive_path = args.target_archive_path
    temp_dir_name = args.temp_dir_name
    compressed_temp_dir_name = args.compressed_temp_dir_name
    
    extract_zip_to_temp(target_archive_path, temp_directory=temp_dir_name)
    images_to_process = get_image_filepaths(temp_dir_name)
    
    processed_images = 0
    for image_path in images_to_process:
        offset = 0.05
        
        #   first we reduce amount of required calculations by making grayscaled images
        # instead of colored ones
        input_img = cv2.imread(image_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        
        #   now we pass grayscaled image to the Haar Cascade and it will return list with
        # all found face coordinates
        faces = face_cascade.detectMultiScale(input_img, scaleFactor=1.1, minNeighbors=10)
        for (x, y, w, h) in faces:            
            processed_images += 1
            #   considering found face, apply offset to save some additional face elements
            x_offset, y_offset = int(w * offset), int(h * offset)
            
            #   considering face start point, its width and height, offsets to apply -
            # calculate start point, end point and make sure that it's not going out
            # of the image size
            x = x - x_offset if x_offset < x else 0
            y = y - y_offset if y_offset < y else 0
            x_end = x + w + 2 * x_offset
            y_end = y + h + 2 * y_offset 
            x_end = x_end if x_end < input_img.shape[1] else input_img.shape[1]
            y_end = y_end if y_end < input_img.shape[0] else input_img.shape[0]
            
            #   extract face and convert face image from cv2 one into Pillow one,
            # considering possible error of having incorrect image mode
            face_img = input_img[y:y_end, x:x_end]           
            face_img = Image.fromarray(face_img)
            if face_img.mode != "RGB":
                face_img = face_img.convert("RGB")
            
            # Save the extracted face as a separate image file
            if not os.path.exists(compressed_temp_dir_name):
                os.makedirs(compressed_temp_dir_name)
            face_img.save(f"{os.getcwd()}/{compressed_temp_dir_name}/face_{processed_images}.jpg", "JPEG")
    #   remove intermediate directory
    remove_temp_directory(temp_dir_name)
    
    #   we want the name of original archive with flag that it's compressed
    base_name, extension = os.path.splitext(target_archive_path)
    new_base_name = base_name + "_compressed"
    new_file_name = new_base_name + extension
    
    #   make new archive with contents of the directory containing compressed images
    with zipfile.ZipFile(new_file_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(compressed_temp_dir_name):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, compressed_temp_dir_name))
    
    #   final removal of intermediate directory with compressed image and original zip archive
    remove_temp_directory(compressed_temp_dir_name)
    if os.path.exists(target_archive_path):
        os.remove(target_archive_path)
        

#   if you want original function, use this or look into final chapter of attached ipynb
# optimize_images_zip_archive("images.zip", "temporary", "compressed")

#   final call will look like "python opencv_images_face_zip_compresser.py images.zip temporary compressed"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Take zip archive from given filename or path in " +
                                                  "'target_archive_path',extract images out of it into" +
                                                  " temporary directory that will be created with " +
                                                  "'temp_dir_name' name, then from all images will be" +
                                                  "taken face and transformed into JPEG format."))
    parser.add_argument("target_archive_path", help="name of archive to compress")
    parser.add_argument("temp_dir_name", help="name of temp dir where to save extracted imgs")
    parser.add_argument("compressed_temp_dir_name", help="name of temp dir where to save compressed imgs")
    
    args = parser.parse_args()
    optimize_images_zip_archive(args)
