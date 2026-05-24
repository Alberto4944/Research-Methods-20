# (index, name) — for readability and CSV column naming
RELEVANT_JOINTS = [
    (0,  "nose"),               # head/body orientation proxy

    (11, "left_shoulder"),      # body rotation reference
    (12, "right_shoulder"),     # swing arm root

    (13, "left_elbow"),         # body rotation reference  
    (14, "right_elbow"),        # swing arm hinge

    (15, "left_wrist"),         # opposite side balance
    (16, "right_wrist"),        # racket hand — most important

    (18, "right_pinky"),        # racket grip finish position
    (20, "right_index"),        # racket grip finish position

    (23, "left_hip"),           # weight transfer + stance
    (24, "right_hip"),          # weight transfer + stance
]

# Just the indices, for easy slicing
JOINT_INDICES = [j[0] for j in RELEVANT_JOINTS]
JOINT_NAMES   = [j[1] for j in RELEVANT_JOINTS]

# Build flat CSV column names: lm_nose_x, lm_nose_y, lm_nose_z, ...
FEATURE_COLS = []
for name in JOINT_NAMES:
    FEATURE_COLS += [f"lm_{name}_x", f"lm_{name}_y", f"lm_{name}_z"]

# Total number of features (11 joints × 3 axes)
N_FEATURES = len(FEATURE_COLS)  # 33

if __name__ == "__main__":
    print(f"Tracking {len(RELEVANT_JOINTS)} joints → {N_FEATURES} features per frame")
    for idx, name in RELEVANT_JOINTS:
        print(f"  [{idx:2d}] {name}")