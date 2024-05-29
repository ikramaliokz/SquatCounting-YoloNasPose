import cv2
import numpy as np
import math
import argparse
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization

# Streamlit App
import streamlit as st
import tempfile
st.title("Squat Counting By YOLO NAS")
start_button = st.button("Start")
frame_placeholder = st.empty()
stop_button = st.button("Stop")


def calculate_angle(key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        # Calculate the slope of two straight lines
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # Convert radians to angles
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff
    # angle = 0
    # if len(key_points)!=0:
    left_points = [[key_points[0][i][0], key_points[0][i][1]] for i in left_points_idx]
    right_points = [[key_points[0][i][0], key_points[0][i][1]] for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
    return angle


def put_text(frame, exercise, count,redio):
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(163 * redio)),
        (55, 104, 0), -1
    )

    if exercise == 'Squat':
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(100 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )

def main():
    # Obtain relevant parameters

    model = models.get(Models.YOLO_NAS_POSE_N, pretrained_weights="coco_pose")

    # Open the video file or camera

    cap = cv2.VideoCapture('squat.mp4')
    # Set variables to record motion status
    reaching = False
    reaching_last = False
    state_keep = False
    counter = 0

    # Loop through the video frames
    while cap.isOpened() and start_button:
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Set plot size redio for inputs with different resolutions
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)
            # Run YOLO-NAS inference on the frame
            results = model.predict(frame, conf=0.50)
            preds = results._images_prediction_lst[0] 
            poses = preds.prediction.poses
            # Preventing errors caused by special scenarios
            if len(poses) == 0:

                # if args.show:
                put_text(frame, 'No Object', counter,plot_size_redio)
                scale = 640 / max(frame.shape[0], frame.shape[1])
                show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                frame_placeholder.image(show_frame, channels="RGB")
                # cv2.imshow("YOLOv8 Inference", show_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Get hyperparameters
            left_points_idx = [11, 13, 15]
            right_points_idx = [12, 14, 16]

            # Calculate angle
            angle = calculate_angle(poses, left_points_idx, right_points_idx)
            # Determine whether to complete once
            if angle < 80:
                reaching = True
            if angle > 140:
                reaching = False

            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                if not reaching and state_keep:
                    counter += 1
                    state_keep = False

            # Visualize the results on the frame
            annotated_frame = PoseVisualization.draw_poses(image=preds.image,poses=poses,
                                                           boxes=preds.prediction.bboxes_xyxy,
                                                           scores=preds.prediction.scores,
                                                           is_crowd=None,edge_links= preds.prediction.edge_links,
                                                           edge_colors=preds.prediction.edge_colors,keypoint_colors=preds.prediction.keypoint_colors)

            # # add relevant information to frame
            put_text(
                annotated_frame, 'Squat', counter, plot_size_redio)
            # # Display the annotated frame
            scale = 640 / max(annotated_frame.shape[0], annotated_frame.shape[1])
            show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
            frame_placeholder.image(show_frame, channels="RGB")
            # cv2.imshow("YOLOv8 Inference", show_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
