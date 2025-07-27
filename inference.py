




# import time


# import cv2
# import mediapipe as mp
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F

# # Constants
# WINDOW_SIZE = 30
# CLASSES = ['Reach to shelf', 'Retract from shelf', 'hand on shelf', 'inspecting']
# TOT_ACTION_CLASSES = len(CLASSES)

# # Initialize MediaPipe Holistic model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# holistic = mp_holistic.Holistic(static_image_mode=False,
#                                  model_complexity=1,
#                                  enable_segmentation=False,
#                                  refine_face_landmarks=False,
#                                  min_detection_confidence=0.5,
#                                  min_tracking_confidence=0.5)

# class ActionClassificationBiLSTM(nn.Module):
#     def __init__(self, input_features, hidden_dim, num_layers=3, learning_rate=0.001):
#         super().__init__()
        
#         # BiLSTM layers
#         self.lstm = nn.LSTM(
#             input_features, 
#             hidden_dim, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             bidirectional=True, 
#             dropout=0.3  # Dropout between stacked LSTM layers
#         )
        
#         # Fully connected layers with batch normalization and dropout
#         self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.batch_norm_fc1 = nn.BatchNorm1d(hidden_dim)
        
#         # Attention mechanism
#         self.attention = nn.Linear(hidden_dim * 2, 1)
        
#         # Final output layer
#         self.fc2 = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)
        
#         # Dropout
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # BiLSTM layers
#         lstm_out, _ = self.lstm(x)  # Output: [batch, seq_len, hidden_dim*2]
        
#         # Attention weights
#         attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        
#         # Weighted sum of LSTM output with attention
#         weighted_lstm_out = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_dim*2]
        
#         # First fully connected layer with batch norm and ReLU
#         x = self.fc1(weighted_lstm_out)
#         x = self.batch_norm_fc1(x)
#         x = torch.relu(x)
        
#         # Dropout
#         x = self.dropout(x)
        
#         # Output layer
#         x = self.fc2(x)
        
#         return x


# # Load the model
# def load_model(model_path):
#     input_features = 44
#     hidden_dim = 64
#     model = ActionClassificationBiLSTM(input_features, hidden_dim)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model


# # Function to process a single video and predict actions
# def process_video_and_predict(video_path, model, keypoints_file_path):
#     with open(keypoints_file_path, 'w') as f:
#         cap = cv2.VideoCapture(video_path)
#         keyframes = []
#         frame_count = 0
#         predicted_action = "Waiting for prediction..."
#         # total_start_time = time.time()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_start_time = time.time()

#             frame_count += 1

#             # Convert BGR frame to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(rgb_frame)

#             keypoints_line = []

#             # Pose keypoints
#             if results.pose_landmarks:
#                 pose_landmarks = results.pose_landmarks.landmark
#                 keypoints = {
#                     'left_shoulder': pose_landmarks[11],
#                     'right_shoulder': pose_landmarks[12],
#                     'left_elbow': pose_landmarks[13],
#                     'right_elbow': pose_landmarks[14],
#                     'left_wrist': pose_landmarks[15],
#                     'right_wrist': pose_landmarks[16],
#                     'left_hip': pose_landmarks[23],
#                     'right_hip': pose_landmarks[24],
#                     'left_knee': pose_landmarks[25],
#                     'right_knee': pose_landmarks[26],
#                     'left_ankle': pose_landmarks[27],
#                     'right_ankle': pose_landmarks[28]
#                 }
#                 for landmark in keypoints.values():
#                     keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
#             else:
#                 #print(f"Frame {frame_count} has no keypoints, skipping.")
#                 keypoints_line.extend(['0.000000,0.000000']*12)

#             # Left hand keypoints
#             if results.left_hand_landmarks:
#                 left_hand_landmarks = results.left_hand_landmarks.landmark
#                 finger_tips = [left_hand_landmarks[4], left_hand_landmarks[8], left_hand_landmarks[12], left_hand_landmarks[16], left_hand_landmarks[20]]
#                 for landmark in finger_tips:
#                     keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
#             else:
#                 #print(f"Frame {frame_count} has no left hand keypoints, skipping.")
#                 keypoints_line.extend(['0.000000,0.000000']*5)

#             # Right hand keypoints
#             if results.right_hand_landmarks:
#                 right_hand_landmarks = results.right_hand_landmarks.landmark
#                 finger_tips = [right_hand_landmarks[4], right_hand_landmarks[8], right_hand_landmarks[12], right_hand_landmarks[16], right_hand_landmarks[20]]
#                 for landmark in finger_tips:
#                     keypoints_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
#             else:
#                 #print(f"Frame {frame_count} has no right hand keypoints, skipping.")
#                 keypoints_line.extend(['0.000000,0.000000']*5)


#             f.write(','.join(keypoints_line) + '\n')
#             keyframes.append([float(kp) for kp in ','.join(keypoints_line).split(',')])
#             print(f"Frame {frame_count} captured.")

#             # Prediction every 30 frames
#             if len(keyframes) == WINDOW_SIZE:
#                 input_tensor = torch.tensor(keyframes, dtype=torch.float32).unsqueeze(0)
#                 with torch.no_grad():
#                     predictions = model(input_tensor)
#                     predicted_class = torch.argmax(predictions, dim=1).item()
#                     predicted_action = CLASSES[predicted_class]

#                 print(f"Frame {frame_count}: Predicted action - {predicted_action}")

#                 # Reset keyframes sequence for the next window
#                 keyframes = []


#             frame_end_time = time.time()

#             total_time = frame_end_time - frame_start_time

#             fps = 1/total_time
#             print(fps)

#             # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)




#             # # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             # # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             # # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#             # # Display the prediction text
#             # cv2.putText(frame, f'Predicted action: {predicted_action}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

#             # # cv2.imshow('Video with Keypoints and Predictions', frame)

#             # frame_resize = cv2.resize(frame, (840, 640))

#             # cv2.imshow('Video with Keypoints and Predictions', frame_resize)

#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#                 #break

#         # cap.release()
#         # cv2.destroyAllWindows()

#         # total_end_time = time.time()
#         # time_taken = total_end_time - total_start_time

#         # print(f"total time taken is {time_taken}")


# MODEL_PATH = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\BiLSTM_64neuron_4class_best_till.pt"
# video_path = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\lab_merl_test_0001.mp4"
# keypoints_file_path = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\keypoints_output.txt"

# model = load_model(MODEL_PATH)
# process_video_and_predict(video_path, model, keypoints_file_path)
# holistic.close()


######################






import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Constants
WINDOW_SIZE = 30
CLASSES = ['Reach to shelf', 'Retract from shelf', 'hand on shelf', 'inspecting']
TOT_ACTION_CLASSES = len(CLASSES)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=False,
                                 model_complexity=1,
                                 enable_segmentation=False,
                                 refine_face_landmarks=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

class ActionClassificationBiLSTM(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers=3, learning_rate=0.001):
        super().__init__()

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3  # Dropout between stacked LSTM layers
        )

        # Fully connected layers with batch normalization and dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm_fc1 = nn.BatchNorm1d(hidden_dim)

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Final output layer
        self.fc2 = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # BiLSTM layers
        lstm_out, _ = self.lstm(x)  # Output: [batch, seq_len, hidden_dim*2]

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]

        # Weighted sum of LSTM output with attention
        weighted_lstm_out = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_dim*2]

        # First fully connected layer with batch norm and ReLU
        x = self.fc1(weighted_lstm_out)
        x = self.batch_norm_fc1(x)
        x = torch.relu(x)

        # Dropout
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x


# Load the model
def load_model(model_path):
    input_features = 44
    hidden_dim = 64
    model = ActionClassificationBiLSTM(input_features, hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to the appropriate device

    model.eval()
    return model


# Function to process a single video and predict actions
def process_video_and_predict(video_path, model, keypoints_file_path):
    with open(keypoints_file_path, 'w') as f:
        cap = cv2.VideoCapture(video_path)
        keyframes = []
        frame_count = 0
        predicted_action = "Waiting for prediction..."
        first_time = time.time()
        

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_start_time = time.time()

            frame_count += 1

            # Convert BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            keyframes_line = []

 
                        # Pose keypoints
            if results.pose_landmarks:
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
                for landmark in keypoints.values():
                    keyframes_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
            else:
                keyframes_line.extend(['0.000000,0.000000']*12)

            # Left hand keypoints
            if results.left_hand_landmarks:
                left_hand_landmarks = results.left_hand_landmarks.landmark
                finger_tips = [left_hand_landmarks[4], left_hand_landmarks[8], left_hand_landmarks[12], left_hand_landmarks[16], left_hand_landmarks[20]]
                for landmark in finger_tips:
                    keyframes_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
            else:
                keyframes_line.extend(['0.000000,0.000000']*5)

            # Right hand keypoints
            if results.right_hand_landmarks:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                finger_tips = [right_hand_landmarks[4], right_hand_landmarks[8], right_hand_landmarks[12], right_hand_landmarks[16], right_hand_landmarks[20]]
                for landmark in finger_tips:
                    keyframes_line.append(f'{landmark.x:.6f},{landmark.y:.6f}')
            else:
                keyframes_line.extend(['0.000000,0.000000']*5)

            f.write(','.join(keyframes_line) + '\n')
            keyframes.append([float(kp) for kp in ','.join(keyframes_line).split(',')])
            print(f"Frame {frame_count} captured.")

            # Prediction every 30 frames
            if len(keyframes) == WINDOW_SIZE:
                input_tensor = torch.tensor(keyframes, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    predictions = model(input_tensor)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    predicted_action = CLASSES[predicted_class]

                print(f"Frame {frame_count}: Predicted action - {predicted_action}")


                

                # Reset keyframes sequence for the next window
                keyframes = []

            frame_end = time.time()


            time_taken_for_each_frame = (frame_end - frame_start_time)
            print(f"time taken for frame no {frame_count} : {time_taken_for_each_frame:.2f}")


            frame_end_time = time.time()
            fps = 1 / (frame_end_time - frame_start_time)
            print(f"FPS: {fps:.2f}")


            
            
            

        cap.release()

        last_time = time.time()
        time_taken = last_time-first_time
        print(f"time taken : {time_taken}")


MODEL_PATH = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\BiLSTM_64neuron_4class_best_till.pt"
video_path = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\lab_merl_test_0001.mp4"
keypoints_file_path = r"D:\gaurav\shopper_mediapipe_handpose\merl_classification\keypoints_output.txt"

model = load_model(MODEL_PATH)
process_video_and_predict(video_path, model, keypoints_file_path)
holistic.close()



                        