import numpy as np

# ── Math helpers ──────────────────────────────────────────────────────────────

def calc_angle(a, b, c):
    """Calculate the angle at point b formed by a->b->c, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def landmark_to_point(lm, dims=2):
    """Convert a MediaPipe landmark to a numpy array. dims=2 for x/y, 3 for x/y/z."""
    if dims == 3:
        return np.array([lm.x, lm.y, lm.z])
    return np.array([lm.x, lm.y])

def normalize_distance(lm_a, lm_b):
    """Return the distance between two landmarks — used as body scale reference."""
    a = landmark_to_point(lm_a)
    b = landmark_to_point(lm_b)
    return np.linalg.norm(a - b) + 1e-6

# ── Feedback rules ────────────────────────────────────────────────────────────
# Each rule returns a string if the issue is detected, or None if form is good.
# Thresholds are normalized relative to shoulder width so they work for any body size.

# SIDE VIEW RULES
# Camera is to the left or right of the player.
# Good for: elbow extension, forward swing arc, weight transfer front/back.

def check_elbow_angle_side(landmarks, scale):
    r_shoulder = landmark_to_point(landmarks[12])
    r_elbow    = landmark_to_point(landmarks[14])
    r_wrist    = landmark_to_point(landmarks[16])
    angle = calc_angle(r_shoulder, r_elbow, r_wrist)
    # Ideal forehand drive elbow angle at contact: 120-160 degrees
    if angle < 110:
        return "Bend your elbow less — arm too cramped"
    if angle > 170:
        return "Bend your elbow more — arm too straight"
    return None

def check_wrist_height_side(landmarks, scale):
    r_shoulder = landmark_to_point(landmarks[12])
    r_wrist    = landmark_to_point(landmarks[16])
    # Wrist should be below shoulder at contact (positive y = lower on screen)
    diff = (r_wrist[1] - r_shoulder[1]) / scale
    if diff < -0.3:
        return "Lower your wrist — too high at contact"
    return None

def check_forward_lean_side(landmarks, scale):
    nose      = landmark_to_point(landmarks[0])
    l_hip     = landmark_to_point(landmarks[23])
    r_hip     = landmark_to_point(landmarks[24])
    hip_mid_x = (l_hip[0] + r_hip[0]) / 2
    # Nose should be ahead of hips (smaller x if facing right)
    diff = (nose[0] - hip_mid_x) / scale
    if abs(diff) < 0.1:
        return "Lean into the shot — transfer your weight forward"
    return None

# FRONT VIEW RULES
# Camera is in front of or behind the player.
# Good for: hip rotation, shoulder symmetry, lateral wrist position.

def check_hip_rotation_front(landmarks, scale):
    l_hip = landmark_to_point(landmarks[23])
    r_hip = landmark_to_point(landmarks[24])
    # On a forehand drive hips should rotate — right hip forward means r_hip.x < l_hip.x
    diff = (l_hip[0] - r_hip[0]) / scale
    if diff < 0.1:
        return "Rotate your hips — turn into the shot"
    return None

def check_shoulder_rotation_front(landmarks, scale):
    l_shoulder = landmark_to_point(landmarks[11])
    r_shoulder = landmark_to_point(landmarks[12])
    diff = (l_shoulder[0] - r_shoulder[0]) / scale
    if diff < 0.05:
        return "Rotate your shoulders — follow through more"
    return None

def check_wrist_position_front(landmarks, scale):
    r_shoulder = landmark_to_point(landmarks[12])
    r_wrist    = landmark_to_point(landmarks[16])
    # Wrist should cross midline on follow through
    diff = (r_shoulder[0] - r_wrist[0]) / scale
    if diff < 0.1:
        return "Follow through across your body more"
    return None

# ── Main feedback function ────────────────────────────────────────────────────

SIDE_RULES  = [check_elbow_angle_side, check_wrist_height_side, check_forward_lean_side]
FRONT_RULES = [check_hip_rotation_front, check_shoulder_rotation_front, check_wrist_position_front]

def get_feedback(landmarks, view):
    """
    Run all feedback rules for the given camera view.

    Args:
        landmarks : list of MediaPipe NormalizedLandmark (result.pose_landmarks[0])
        view      : "side" or "front"

    Returns:
        list of feedback strings (empty list = good form)
    """
    # Use shoulder width as the body scale reference
    scale = normalize_distance(landmarks[11], landmarks[12])
    rules = SIDE_RULES if view == "side" else FRONT_RULES

    tips = []
    for rule in rules:
        result = rule(landmarks, scale)
        if result:
            tips.append(result)
    return tips


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("feedback.py loaded ok")
    print(f"Side rules  : {len(SIDE_RULES)}")
    print(f"Front rules : {len(FRONT_RULES)}")
    print("Import get_feedback and pass it a pose_landmarks[0] list to use.")