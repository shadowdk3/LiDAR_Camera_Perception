#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

class LidarImageFusion : public rclcpp::Node
{
    public:
        LidarImageFusion() : Node("lidar_image_funsion_node")
        {
            this->declare_parameter("show_unmatched_tracks", false);                            // Configures whether to display LiDAR tracks that could not be matched with any YOLO detections
            show_unmatched_tracks_ = this->get_parameter("show_unmatched_tracks").as_bool();    // example: 

            this->declare_parameter("overlap_threshold", 0.15);                                 // Minimum 2D IoU overlap required to consider a LiDAR track and a YOLO detection as the same object
            overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();

            loadLidarCameraCalibration();

            subscription_camera_object_detections_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
                "/camera/object_detections", 10, std::bind(&LidarImageFusion::yolo_object_detection_callback, this, std::placeholders::_1));
            
            subscription_lidar_tracked_object_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
                "/lidar_track/tracked_objects", 10, std::bind(&LidarImageFusion::lidar_tracked_object_callback, this, std::placeholders::_1));
            
            subscription_left_color_camera_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/kitti/image/color/left", 10, std::bind(&LidarImageFusion::image_callback, this, std::placeholders::_1));

            publisher_fusion_image_ = this->create_publisher<sensor_msgs::msg::Image>("/fusion/identified_objects", 10);

            RCLCPP_INFO(this->get_logger(), "Image Overlay Node Started...");
        }

    private:
        bool show_unmatched_tracks_;
        double overlap_threshold_;
        cv::Mat velo_to_cam2_projection_; 

        vision_msgs::msg::Detection2DArray latest_yolo_detections_;
        std::vector<visualization_msgs::msg::Marker> latest_tracks_;

        rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr subscription_camera_object_detections_;
        rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr subscription_lidar_tracked_object_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_left_color_camera_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_fusion_image_;

        void loadLidarCameraCalibration()
        {
            // Camera define in KITTI
            // 0: left gray
            // 1: right gary
            // 2: left color
            // 3: right color
            
            // Extrinsic matrix from LiDAR (Velodyne) to left gray camera reference frame
            // KITTI calibration file: calib_velo_to_cam.txt
            //  Homogeneous Transformation Matrix = [R | T]
            //      Rotation Matrix (R)            |   Translation (T)
            // 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03
            // 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 
            // 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
            cv::Mat velo_to_cam0_extrinsic = (cv::Mat_<double>(4, 4) <<
                7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
                1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02,
                9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01,
                0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00);

            // KITTI calibration file: calib_cam_to_cam.txt
            // Camera Rectification Matrix: left gary camera, R_rect_00 (4x4)
            // Formed by extending the 3x3 rotation matrix into a 4x4 identity framework
            //       Rotation Matrix (R)          | Translation (0) 
            // 9.999239e-01 9.837760e-03 -7.445048e-03 0.000000e+00 
            // -9.869795e-03 9.999421e-01 -4.278459e-03 0.000000e+00 
            // 7.402527e-03 4.351614e-03 9.999631e-01 0.000000e+00 
            // 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 
            cv::Mat cam0_rectification = (cv::Mat_<double>(4, 4) <<
                 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00,
                -9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00,
                 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00);

            // KITTI calibration file: calib_cam_to_cam.txt
            // Projection Rectified Matrix for left color camera, P_rect_02 (3x4)
            // P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 
            //            0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 
            //            0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
            cv::Mat cam2_projection_rectified = (cv::Mat_<double>(3, 4) <<
                7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
                0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03);

            // Pre-compute the unified 3D-to-2D projection matrix for massive speedup.
            // Coordinate transformation chain: LiDAR (Velodyne) -> Cam0 -> Rectified Cam0 -> Cam2 Pixels
            velo_to_cam2_projection_ = cam2_projection_rectified * cam0_rectification * velo_to_cam0_extrinsic;
        }

        // This caches the latest visual detections asynchronously, which are subsequently 
        // processed and fused with 3D LiDAR tracks inside the image_callback thread.
        void yolo_object_detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
        {
            latest_yolo_detections_ = *msg;
        }

        void lidar_tracked_object_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
        {
            latest_tracks_.clear(); // clear last record

            // push all detected track in the list
            for (const auto& marker : msg->markers) 
            {
                if (marker.type == visualization_msgs::msg::Marker::CUBE && marker.action == visualization_msgs::msg::Marker::ADD) {
                    latest_tracks_.push_back(marker);
                }
            }
        }

        /**
         * @brief Projects a 3D LiDAR spatial coordinate into 2D camera image pixel coordinates.
         * @param x 3D point location in meters along the sensor's X-axis (Forward in KITTI setup).
         * @param y 3D point location in meters along the sensor's Y-axis (Left in KITTI setup).
         * @param z 3D point location in meters along the sensor's Z-axis (Up in KITTI setup).
         * @return cv::Point2f The designated (u, v) pixel coordinate, or (-1, -1) if the point lies behind the camera lens.
         */
        cv::Point2f project3DTo2D(double x, double y, double z) 
        {
            // Construct a 4x1 homogeneous column vector by appending a scaling factor of 1.0…
            cv::Mat pt_3d = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);

            // Apply the unified projection matrix chain: [K * R_rect * Extrinsics] * P_3ds
            cv::Mat pt_2d = velo_to_cam2_projection_ * pt_3d;
            
            // Extract the scale factor 'depth_w', which represents the physical depth along the camera's optical axis
            double depth_w = pt_2d.at<double>(2, 0);
            if (depth_w <= 0) return cv::Point2f(-1, -1);
            
            // Behind-the-camera gating: Eliminate points that fall behind the image plane frustum
            if (depth_w <= 0.0) {
                return cv::Point2f(-1.0f, -1.0f);
            }
            
            // Perform perspective division to obtain normalized, axis-aligned (u, v) image plane coordinates
            float pixel_u = static_cast<float>(pt_2d.at<double>(0, 0) / depth_w);
            float pixel_v = static_cast<float>(pt_2d.at<double>(1, 0) / depth_w);

            return cv::Point2f(pixel_u, pixel_v);
        }

        /**
         * @brief Computes the Intersection over Union (IoU) overlap metric between two 2D bounding boxes.
         * @param box_a First boundary rectangle (e.g., Projected 3D MOT track box).
         * @param box_b Second boundary rectangle (e.g., Camera 2D YOLO detection box).
         * @return double The association score between 0.0 (no overlap) and 1.0 (perfect geometric alignment).
         */
        double calculate_iou(const cv::Rect2f& boxA, const cv::Rect2f& boxB) 
        {
            // Determine the intersecting overlap rectangle using OpenCV's bitwise AND operator
            cv::Rect2f intersection = boxA & boxB;
            double intersection_area  = intersection.area();

            // Compute the total combined union area using the principle of inclusion-exclusion
            double union_area = boxA.area() + boxB.area() - intersection_area ;

            // Zero-division protection: Prevent runtime arithmetic exceptions if inputs are degenerate
            if (union_area <= 0) 
                return 0.0;

            // Return the standard Jaccard index (Intersection over Union ratio)
            return intersection_area  / union_area;
        }

        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
        {
            // Convert ROS2 image to CV image
            cv_bridge::CvImagePtr cv_ptr;
            try 
            {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            } 
            catch (cv_bridge::Exception& e) 
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            cv::Mat img = cv_ptr->image;

            // Parse YOLO Detections
            std::vector<cv::Rect2f> yolo_boxes;
            std::vector<std::string> yolo_labels;
            for (const auto& yolo_detection  : latest_yolo_detections_.detections) 
            {
                // ensure the classification results array is not empty to prevent runtime crashes
                if (yolo_detection.results.empty()) 
                {
                    RCLCPP_WARN(this->get_logger(), "Encountered an empty YOLO classification result slot. Skipping.");
                    continue;
                }

                // Calculate the top-left vertex coordinates from center-mass positions
                float top_left_x = yolo_detection.bbox.center.position.x - yolo_detection.bbox.size_x / 2.0;
                float top_left_y = yolo_detection.bbox.center.position.y - yolo_detection.bbox.size_y / 2.0;

                yolo_boxes.push_back(cv::Rect2f(top_left_x, top_left_y, yolo_detection.bbox.size_x, yolo_detection.bbox.size_y));
                yolo_labels.push_back(yolo_detection.results[0].hypothesis.class_id);
            }

            // Sort tracks object by distance (furthest to closest for proper drawing occlusion)
            std::sort(latest_tracks_.begin(), latest_tracks_.end(),
                [](const visualization_msgs::msg::Marker& a, const visualization_msgs::msg::Marker& b) 
            {
                return a.pose.position.x > b.pose.position.x;
            });

            // Setup valid tracking target collections
            std::vector<visualization_msgs::msg::Marker> valid_tracks;
            std::vector<cv::Rect2f> valid_mot_boxes;
            std::vector<double> distances;

            // Pre-allocate memory based on the track list size to eliminate vector reallocation overhead
            valid_tracks.reserve(latest_tracks_.size());
            valid_mot_boxes.reserve(latest_tracks_.size());
            distances.reserve(latest_tracks_.size());

            // Loop through each active 3D MOT (Multi-Object Tracking) track to project its 3D volume into 2D image space
            for (const auto& track : latest_tracks_) 
            {
                const auto& pos = track.pose.position;
                const auto& dim = track.scale;
                
                // Calculate the real-world 3D Euclidean distance (range) from the ego-vehicle sensor origin
                double range_distance = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

                // Generate the 8 vertices of the 3D bounding box using local dimension offsets
                std::vector<cv::Point2f> projected_2d_corners;
                projected_2d_corners.reserve(8); // A 3D box always yields exactly 8 corner coordinates

                double dx_offsets[] = {-dim.x / 2.0, dim.x / 2.0};
                double dy_offsets[] = {-dim.y / 2.0, dim.y / 2.0};
                double dz_offsets[] = {-dim.z / 2.0, dim.z / 2.0};

                // 3-layer nested loop to permute and calculate all 8 distinct geometric corner combinations
                for (double dx : dx_offsets) 
                {
                    for (double dy : dy_offsets) 
                    {
                        for (double dz : dz_offsets) 
                        {
                            // Project the 3D spatial coordinate into 2D image pixel coordinates (u, v)
                            cv::Point2f pixel_point = project3DTo2D(pos.x + dx, pos.y + dy, pos.z + dz);
                            
                            // Validate that the point lies within a valid forward projection frustum (p.x != -1)
                            if (pixel_point.x != -1) {
                                projected_2d_corners.push_back(pixel_point);
                            }
                        }
                    }
                }

                // Geometric gating: Reject the track if it lacks sufficient valid vertices to construct a 2D profile
                if (projected_2d_corners.size() < 4) 
                    continue;

                // Extract the minimum axis-aligned 2D bounding box from 3D projected corners
                cv::Rect2f mot_2d_box = cv::boundingRect(projected_2d_corners);
                
                // Save the synchronized spatial and structural information for downstream data association
                valid_tracks.push_back(track);
                valid_mot_boxes.push_back(mot_2d_box);
                distances.push_back(range_distance);
            }

            // Greedy Matching Algorithm (Fast alternative to Hungarian)
            // Compute pairwise overlap metrics between cross-modal boundaries
            std::map<int, int> matched_mot_indices; // mot_index -> yolo_index
            
            struct Match {
                int mot_idx; 
                int yolo_idx; 
                double iou; 
            };

            std::vector<Match> potential_matches;

            // Generate a sparse cost matrix of all valid geometric 2D IoU overlaps
            for (size_t i = 0; i < valid_mot_boxes.size(); ++i) 
            {
                for (size_t j = 0; j < yolo_boxes.size(); ++j) 
                {
                    double iou = calculate_iou(valid_mot_boxes[i], yolo_boxes[j]);

                    // Gating criteria: Only retain associations that exceed the minimum spatial overlap threshold
                    if (iou > overlap_threshold_) 
                    {
                        potential_matches.push_back({(int)i, (int)j, iou});
                    }
                }
            }

            // Sort matches by highest IoU first
            // Prioritize highest overlap associations first to maximize true positive matches
            std::sort(potential_matches.begin(), potential_matches.end(),
                [](const Match& a, const Match& b) { return a.iou > b.iou; });

            // Track which YOLO detection boxes have already been assigned
            std::vector<bool> yolo_used(yolo_boxes.size(), false);

            for (const auto& match : potential_matches) 
            {
                // Accept the match only if both the MOT track and the YOLO detection are still unassigned
                if (matched_mot_indices.find(match.mot_idx) == matched_mot_indices.end() && !yolo_used[match.yolo_idx]) 
                {
                    matched_mot_indices[match.mot_idx] = match.yolo_idx;        // Confirm the match and link the MOT track index to the YOLO detection index
                    yolo_used[match.yolo_idx] = true;                           // Mark this YOLO detection as used to prevent duplicate assignments
                }
            }

            // Visualization
            for (size_t i = 0; i < valid_tracks.size(); ++i) {
                cv::Rect2f draw_box;
                std::string best_label;
                cv::Scalar color;
                int thickness;

                if (matched_mot_indices.count(i)) {
                    // Match Found
                    int yolo_idx = matched_mot_indices[i];
                    draw_box = yolo_boxes[yolo_idx];
                    best_label = yolo_labels[yolo_idx];
                    color = cv::Scalar(255, 255, 0);
                    thickness = 2;
                } else {
                    // No Match
                    if (!show_unmatched_tracks_) continue;
                    draw_box = valid_mot_boxes[i];
                    best_label = "Object";
                    color = cv::Scalar(150, 150, 150);
                    thickness = 1;
                }
                
                cv::rectangle(img, draw_box, color, thickness);
                char label_text[100];
                std::transform(best_label.begin(), best_label.end(), best_label.begin(), ::toupper);
                snprintf(label_text, sizeof(label_text), "%s | ID:%d | %.1fm", best_label.c_str(), valid_tracks[i].id / 2, distances[i]);
                
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                
                cv::rectangle(img, 
                    cv::Point(draw_box.x, draw_box.y - text_size.height - 5),
                    cv::Point(draw_box.x + text_size.width, draw_box.y),
                    color, cv::FILLED);
                    
                cv::putText(img, label_text, cv::Point(draw_box.x, draw_box.y - 5), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
            }

            publisher_fusion_image_->publish(*cv_bridge::CvImage(msg->header, "bgr8", img).toImageMsg());
        }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarImageFusion>());
    rclcpp::shutdown();
    return 0;
}