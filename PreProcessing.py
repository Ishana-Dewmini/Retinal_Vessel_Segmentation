import cv2
import os
# from albumentations import ElasticTransform, GridDistortion, OpticalDistortion

def resize_images(input_dir, output_dir, target_size,file_type,augmentation):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(file_type):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path) 
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            output_path = os.path.join(output_dir,  f"i_{filename[:-4]}.png")
            cv2.imwrite(output_path, img)

            if augmentation==True:
                # Flip along the x-axis and save
                x_flipped = cv2.flip(img, 0)
                x_flipped_path = os.path.join(output_dir, f"ii_{filename[:-4]}.png")
                cv2.imwrite(x_flipped_path, x_flipped)

                # Flip along the y-axis and save
                y_flipped = cv2.flip(img, 1)
                y_flipped_path = os.path.join(output_dir, f"iii_{filename[:-4]}.png")
                cv2.imwrite(y_flipped_path, y_flipped)

                # Rotate 180 degrees clockwise and save
                rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
                rotated_180_path = os.path.join(output_dir, f"iv_{filename[:-4]}.png")
                cv2.imwrite(rotated_180_path, rotated_180)

def resize_masks(input_dir, output_dir, target_size,augmentation):
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
                output_path = os.path.join(output_dir, f"i_{filename[:-4]}.png")  # Save as PNG
                cv2.imwrite(output_path, resized_frame)

                if augmentation==True:
                    # Flip along the x-axis and save
                    x_flipped = cv2.flip(resized_frame, 0)
                    x_flipped_path = os.path.join(output_dir, f"ii_{filename[:-4]}.png")
                    cv2.imwrite(x_flipped_path, x_flipped)

                    # Flip along the y-axis and save
                    y_flipped = cv2.flip(resized_frame, 1)
                    y_flipped_path = os.path.join(output_dir, f"iii_{filename[:-4]}.png")
                    cv2.imwrite(y_flipped_path, y_flipped)

                    # Rotate 180 degrees clockwise and save
                    rotated_180 = cv2.rotate(resized_frame, cv2.ROTATE_180)
                    rotated_180_path = os.path.join(output_dir, f"iv_{filename[:-4]}.png")
                    cv2.imwrite(rotated_180_path, rotated_180)

                    





# Usage example
target_size = (512, 512,)
input_images_dir = 'DRIVE/training/images'
output_images_dir = 'Data/training/images'
resize_images(input_images_dir, output_images_dir, target_size,".tif",True)

input_masks_dir = 'DRIVE/training/1st_manual'
output_masks_dir = 'Data/training/1st_manual'
resize_masks(input_masks_dir, output_masks_dir, target_size,True)



input_images_dir = 'DRIVE/test/images'
output_images_dir = 'Data/test/images'
resize_images(input_images_dir, output_images_dir, target_size,".tif",False)

input_masks_dir = 'DRIVE/test/1st_manual'
output_masks_dir = 'Data/test/1st_manual'
resize_masks(input_masks_dir, output_masks_dir, target_size,False)

