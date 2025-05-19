import numpy as np

# Helper: Compute the angle (in degrees) at joint B formed by points A-B-C
def _angle_between_points(A, B, C):
    # Convert landmarks A, B, C to numpy arrays (assuming each has .x, .y, .z attributes)
    A = np.array([A.x, A.y, A.z])
    B = np.array([B.x, B.y, B.z])
    C = np.array([C.x, C.y, C.z])
    BA = A - B
    BC = C - B
    # Compute angle between BA and BC vectors
    cos_angle = np.dot(BA, BC) / ((np.linalg.norm(BA) * np.linalg.norm(BC)) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # ensure within valid range
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def get_corrections(move, landmarks):
    """Return a list of correction feedback strings for the given move based on pose landmarks."""
    feedback = []
    # Define shortcuts for key landmarks by index (using MediaPipe Pose indices):contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}
    # e.g., left shoulder=11, right shoulder=12, left elbow=13, right elbow=14, left wrist=15, right wrist=16, left hip=23, right hip=24, left knee=25, right knee=26, left ankle=27, right ankle=28.
    ls = landmarks[11]; rs = landmarks[12]  # shoulders
    le = landmarks[13]; re = landmarks[14]  # elbows
    lw = landmarks[15]; rw = landmarks[16]  # wrists
    lh = landmarks[23]; rh = landmarks[24]  # hips
    lk = landmarks[25]; rk = landmarks[26]  # knees
    la = landmarks[27]; ra = landmarks[28]  # ankles

    # Check form for each move
    if move == "jab":
        # Jab: left arm punches, right arm is guard
        # 1. Left arm should be fully extended (elbow nearly straight)
        angle_left_elbow = _angle_between_points(ls, le, lw)
        if angle_left_elbow < 160:
            feedback.append("Extend your left arm fully on the jab")
        # 2. Right hand (guard) should stay up (near head level)
        if rw.y > rs.y:
            feedback.append("Keep your right hand up to guard")

    elif move == "cross":
        # Cross: right arm punches, left arm is guard
        angle_right_elbow = _angle_between_points(rs, re, rw)
        if angle_right_elbow < 160:
            feedback.append("Extend your right arm fully on the cross")
        if lw.y > ls.y:
            feedback.append("Keep your left hand up to guard")

    elif move == "left_hook":
        # Left hook: left arm punches in a horizontal arc, right arm guards
        angle_left_elbow = _angle_between_points(ls, le, lw)
        if angle_left_elbow > 150:
            feedback.append("Bend your left arm (don't throw a hook with a straight arm)")
        if le.y > ls.y:
            feedback.append("Raise your left elbow to shoulder level for the hook")
        if rw.y > rs.y:
            feedback.append("Keep your right hand up to guard")

    elif move == "right_hook":
        # Right hook: right arm punches, left arm guards
        angle_right_elbow = _angle_between_points(rs, re, rw)
        if angle_right_elbow > 150:
            feedback.append("Bend your right arm (don't throw a hook with a straight arm)")
        if re.y > rs.y:
            feedback.append("Raise your right elbow to shoulder level for the hook")
        if lw.y > ls.y:
            feedback.append("Keep your left hand up to guard")

    elif move == "uppercut":
        # Uppercut: assuming right uppercut (right arm punches upward, left guards)
        angle_right_elbow = _angle_between_points(rs, re, rw)
        if angle_right_elbow > 130:
            feedback.append("Bend your right arm more for the uppercut")
        # Check right elbow tuck (should not flare out to the side)
        angle_shoulder = _angle_between_points(re, rs, rh)
        if angle_shoulder > 60:
            feedback.append("Tuck your right elbow closer to your body on the uppercut")
        if lw.y > ls.y:
            feedback.append("Keep your left hand up to guard")

    elif move == "kick":
        # Kick: assuming right leg kick (e.g., front kick), both arms guard
        angle_right_knee = _angle_between_points(rh, rk, ra)
        if angle_right_knee < 150:
            feedback.append("Extend your right leg more for the kick")
        # Both hands should stay up during a kick
        if (lw.y > ls.y) and (rw.y > rs.y):
            feedback.append("Keep your hands up while kicking")

    return feedback
