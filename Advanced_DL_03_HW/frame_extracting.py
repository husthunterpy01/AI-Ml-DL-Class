import cv2
import os
# Function to extract frames
def FrameCapture(path, output_folder):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    os.makedirs(output_folder, exist_ok=True)

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        output_path = os.path.join(output_folder,"frame%d.jpg" % count)
        # Saves the frames with frame-count
        cv2.imwrite(output_path , image)

        count += 1


# Driver Code
if __name__ == '__main__':
    # Calling the function
    output_folder = "./football/Match_2023_3_0_subclip/images"
    FrameCapture("./football/Match_2023_3_0_subclip/Match_2023_3_0_subclip.mp4", output_folder)
