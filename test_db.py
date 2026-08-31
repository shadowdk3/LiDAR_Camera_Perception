#!/usr/bin/env python3
import psycopg2


"""
Database Connection (psycopg2.connect): It logs into your local PostgreSQL instance using the lidar_perception database, 
postgres user, and your new password 123456.PostGIS WKT 

Polygon Creation: It takes your YOLO standard center coordinates (center_x, center_y, width, height) and calculates 
the 4 corner points of the rectangle. It formats them into a Well-Known Text (WKT) string: 
'POLYGON((x1 y1, x2 y1, x2 y2, x1 y2, x1 y1))'.

Safe SQL Insertion: It uses parameterized placeholders (%s) to securely inject ROS2 10 FPS time data (sec, nanosec) 
and the geometry string into the camera_yolo_detections table.

Data Verification: It immediately queries the database using the newly generated inserted_id and pulls the geometry data 
back into readable text using ST_AsText()
"""

def connect_and_insert():
    # Database Configuration Details
    db_config = {
        "host": "localhost",
        "database": "lidar_perception",
        "user": "postgres",
        "password": "123456"
    }

    try:
        # Establish Connection and Create a Cursor
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print("Successfully connected to the PostgreSQL database!")

        # Simulate YOLO Detection Data (One single frame from a 10 FPS sequence)
        sec = 1787768122
        nanosec = 89587933
        frame_id = "base_link"
        class_name = "car"
        confidence = 0.95
        center_x = 621.0
        center_y = 187.5
        width = 120.0
        height = 80.0

        # Calculate the 4 corners of the PostGIS Polygon
        # (Top-Left, Top-Right, Bottom-Right, Bottom-Left, closing back at Top-Left)
        x1, y1 = center_x - width/2, center_y - height/2
        x2, y2 = center_x + width/2, center_y + height/2
        wkt_polygon = f'POLYGON(({x1} {y1}, {x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}))'

        # Define SQL Insert Query
        insert_query = """
            INSERT INTO camera_yolo_detections 
            (sec, nanosec, frame_id, class_name, confidence, center_x, center_y, width, height, bbox_2d_polygon)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_PolygonFromText(%s, 0))
            RETURNING id;
        """

        # Execute Data Insertion
        cursor.execute(insert_query, (
            sec, nanosec, frame_id, class_name, confidence,
            center_x, center_y, width, height, wkt_polygon
        ))
        
        # Retrieve the auto-generated row ID
        inserted_id = cursor.fetchone()[0]
        print(f"Data successfully inserted! Generated Row ID: {inserted_id}")

        # CRITICAL: Commit the transaction to permanently save changes to disk
        conn.commit()

        # Read back the inserted data immediately to verify success
        print("\n🔍 Fetching the inserted object from the database...")
        select_query = """
            SELECT id, sec, nanosec, class_name, confidence, ST_AsText(bbox_2d_polygon) 
            FROM camera_yolo_detections 
            WHERE id = %s;
        """
        cursor.execute(select_query, (inserted_id,))
        row = cursor.fetchone()
        
        if row:
            print("-" * 50)
            print(f"ID: {row[0]}")
            print(f"Timestamp (sec.nanosec): {row[1]}.{row[2]}")
            print(f"Object Label: {row[3]} (Confidence: {row[4]})")
            print(f"PostGIS Geometry Polygon: {row[5]}")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
        if 'conn' in locals():
            conn.rollback()  # Rollback any uncommitted changes if an error happens
    finally:
        # Close cursor and connection to release system resources
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed securely.")

if __name__ == "__main__":
    connect_and_insert()