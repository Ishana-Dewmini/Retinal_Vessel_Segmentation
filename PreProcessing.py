import cv2
import os

def resize_images(input_dir, output_dir, target_size,file_type):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(file_type):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path) 
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)


def resize_masks(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.gif'):
            gif_path = os.path.join(input_dir, filename)
            
            # Read the GIF using OpenCV, which returns a list of frames
            gif_frames = cv2.VideoCapture(gif_path)

            frame_count = 0
            while True:
                ret, frame = gif_frames.read()
                if not ret:
                    break
                
                # Resize each frame individually
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                
                # Increment frame_count before constructing the filename
                frame_count += 1

                # Save the resized frame with the updated frame_count
                output_path = os.path.join(output_dir, f"{filename[:-4]}.png")  # Save as PNG
                cv2.imwrite(output_path, resized_frame)





# Usage example
target_size = (512, 512,)
input_images_dir = 'DRIVE/training/images'
output_images_dir = 'Resized/training/images'
resize_images(input_images_dir, output_images_dir, target_size,".tif")

input_masks_dir = 'DRIVE/training/1st_manual'
output_masks_dir = 'Resized/training/1st_manual'
resize_masks(input_masks_dir, output_masks_dir, target_size)

input_masks_dir = 'DRIVE/training/mask'
output_masks_dir = 'Resized/training/mask'
resize_masks(input_masks_dir, output_masks_dir, target_size)

input_images_dir = 'DRIVE/test/images'
output_images_dir = 'Resized/test/images'
resize_images(input_images_dir, output_images_dir, target_size,".tif")

input_masks_dir = 'DRIVE/test/mask'
output_masks_dir = 'Resized/test/mask'
resize_masks(input_masks_dir, output_masks_dir, target_size)

