        if latest_result and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                mp.tasks.python.vision.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=pose_landmarks,
                    connections=mp.tasks.python.vision.PoseLandmarker.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.tasks.python.vision.drawing_styles.get_default_pose_landmarks_style()
                )