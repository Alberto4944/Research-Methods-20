if latest_result and latest_result.pose_landmarks:
        for pose_landmarks in latest_result.pose_landmarks:
            drawing_utils.draw_landmarks( # Draws all 33 landmarks
                image=annotated,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4)
            )
            all_current_landmarks = np.array([])
            for landmark in pose_landmarks:
                all_current_landmarks = np.append(all_current_landmarks, [landmark.x, landmark.y, landmark.z])
            if (dataset.size > 0):
                dataset = np.vstack((dataset, all_current_landmarks))
            else:
                dataset = all_current_landmarks