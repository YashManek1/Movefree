class Navigator:
    def __init__(self, width):
        self.width = width
        # Define Zones
        self.left_bound = width * 0.35
        self.right_bound = width * 0.65

    def analyze(self, detections):
        """
        Returns: (State, Message)
        State: SAFE, WARNING, STOP
        """
        min_dist = 99.9
        closest_obj = None

        # Zone Clearances
        left_clear = True
        center_clear = True
        right_clear = True

        for det in detections:
            label, dist, bbox = det
            cx = (bbox[0] + bbox[2]) / 2

            # Find closest object generally
            if dist < min_dist:
                min_dist = dist
                closest_obj = label

            # Check Zones
            if dist < 2.5:  # Only care about things closer than 2.5m
                if cx < self.left_bound:
                    left_clear = False
                elif cx > self.right_bound:
                    right_clear = False
                else:
                    center_clear = False

        # --- LOGIC DECISION TREE ---

        # 1. IMMEDIATE DANGER
        if not center_clear and min_dist < 1.2:
            return "STOP", f"Stop! {closest_obj} ahead!"

        # 2. OBSTACLE AHEAD - NAVIGATE
        if not center_clear:
            if left_clear and right_clear:
                return "WARNING", f"{closest_obj} ahead. Go Left or Right."
            elif left_clear:
                return "WARNING", "Obstacle ahead. Turn Left."
            elif right_clear:
                return "WARNING", "Obstacle ahead. Turn Right."
            else:
                return "STOP", "Path blocked completely."

        # 3. SIDE HAZARDS
        if not left_clear:
            return "CAUTION", "Obstacle on left."
        if not right_clear:
            return "CAUTION", "Obstacle on right."

        return "SAFE", "Path Clear"
