
import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                enable_segmentation=False,
                                refine_face_landmarks=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Function to process a single video
def process_video(video_path, output_folder, action_name, series):
    # Create the full path for output folder
    action_folder = os.path.join(output_folder, f'action_{action_name}')
    os.makedirs(action_folder, exist_ok=True)
    
    # Use the same name for the output keypoints file as the video file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_txt_path = os.path.join(action_folder, f'{video_name}.txt')
    
    # Open file to save keypoints
    output_file = open(output_txt_path, 'w')

    # Capture video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the holistic model
        results = holistic.process(rgb_frame)

        keypoints_line = []  # To store all keypoints (pose + selected hand keypoints)
        detection_made = False  # Flag to indicate if any detection is made

        # Extract relevant keypoints from the pose
        if results.pose_landmarks:
            detection_made = True
            pose_landmarks = results.pose_landmarks.landmark
            keypoints = {
                'left_shoulder': pose_landmarks[11],
                'right_shoulder': pose_landmarks[12],
                'left_elbow': pose_landmarks[13],
                'right_elbow': pose_landmarks[14],
                'left_wrist': pose_landmarks[15],
                'right_wrist': pose_landmarks[16],
                'left_hip': pose_landmarks[23],
                'right_hip': pose_landmarks[24],
                'left_knee': pose_landmarks[25],
                'right_knee': pose_landmarks[26],
                'left_ankle': pose_landmarks[27],
                'right_ankle': pose_landmarks[28]
            }
            # Append pose keypoints (body) to the line and draw on frame
            for name, landmark in keypoints.items():
                keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
                # Draw pose keypoints on the video frame
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 255, 0), -1)

        else:
            keypoints_line.extend(['0.000000,0.000000'] * 12)

        # Extract one keypoint per finger for the left hand
        if results.left_hand_landmarks:
            detection_made = True
            left_hand_landmarks = results.left_hand_landmarks.landmark
            # Select the tip of each finger
            finger_tips = [left_hand_landmarks[4], left_hand_landmarks[8], left_hand_landmarks[12], left_hand_landmarks[16], left_hand_landmarks[20]]
            for landmark in finger_tips:
                keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
                # Draw hand keypoints on the video frame
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 0, 255), -1)
        else:
            keypoints_line.extend(['0.000000,0.000000'] * 5)

        # Extract one keypoint per finger for the right hand
        if results.right_hand_landmarks:
            detection_made = True
            right_hand_landmarks = results.right_hand_landmarks.landmark
            # Select the tip of each finger
            finger_tips = [right_hand_landmarks[4], right_hand_landmarks[8], right_hand_landmarks[12], right_hand_landmarks[16], right_hand_landmarks[20]]
            for landmark in finger_tips:
                keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
                # Draw hand keypoints on the video frame
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (255, 0, 0), -1)
        else:
            keypoints_line.extend(['0.000000,0.000000'] * 5)

        # Write to file only if any keypoints were detected
        if detection_made:
            output_file.write(','.join(keypoints_line) + '\n')

        # Display the video with keypoints
        cv2.imshow('Video with Keypoints', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the video capture and output file
    cap.release()
    output_file.close()
    cv2.destroyAllWindows()

# Traverse folder structure and process all videos
def process_videos_in_folders(input_folder, output_folder):
    series_counter = {}  # Dictionary to keep track of video series for each action

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                
                # Extract action name from folder name or another method you define
                action_name = os.path.basename(root)  # Assuming folder name corresponds to the action
                
                # Get or initialize the series counter for this action
                series_counter[action_name] = series_counter.get(action_name, 0) + 1
                series = series_counter[action_name]
                
                print(f'Processing: {video_path} -> Action: {action_name}, Series: {series}')
                
                # Process video and save keypoints
                process_video(video_path, output_folder, action_name, series)

# Main input folder containing 75 subfolders
input_folder = r'C:\Users\admin\Desktop\Merl\output_directory'
output_folder = r'C:\Users\admin\Desktop\mediapipe\merl_related\keypoint_26toAll'  # New folder for storing keypoint text files

# Process videos
process_videos_in_folders(input_folder, output_folder)

# Release the holistic model resources
holistic.close()
