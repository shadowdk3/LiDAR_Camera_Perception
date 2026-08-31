import psycopg2

class DatabaseLogger:
    def __init__(self, logger):
        self.logger = logger  # Pass the ROS node logger for centralized logging
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host="localhost",
                database="lidar_perception",
                user="postgres",
                password="123456"
            )
            self.cursor = self.conn.cursor()
            self.logger.info("DatabaseLogger: Connected to PostgreSQL successfully.")
        except Exception as e:
            self.logger.error(f"DatabaseLogger Connection Error: {e}")
            raise e

    def log_detection(self, sec, nanosec, frame_id, class_name, confidence, center_x, center_y, width, height):
        try:
            # Generate the 2D bounding box polygon envelope coordinates
            x1, y1 = center_x - width / 2, center_y - height / 2
            x2, y2 = center_x + width / 2, center_y + height / 2
            wkt_polygon = f'POLYGON(({x1} {y1}, {x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}))'

            sql = """
                INSERT INTO camera_yolo_detections 
                (sec, nanosec, frame_id, class_name, confidence, center_x, center_y, width, height, bbox_2d_polygon)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_PolygonFromText(%s, 0));
            """
            self.cursor.execute(sql, (
                sec, nanosec, frame_id, class_name, confidence,
                center_x, center_y, width, height, wkt_polygon
            ))
        except Exception as e:
            self.logger.error(f"DatabaseLogger Insert Error: {e}")
            self.conn.rollback()

    def commit_frame(self):
        """Call this once at the end of the frame processing to safely flush records to disk."""
        if self.conn:
            self.conn.commit()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            self.logger.info("DatabaseLogger connection closed safely.")