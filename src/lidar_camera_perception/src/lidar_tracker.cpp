#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

class Kalman_Track {
    public:
        int track_id;                       // Unique identifier assigned to this tracked object
        int frames_cnt;                     // Total number of frames this track has existed since birth
        int match_cnt;                      // Total number of times this track has successfully matched with a sensor detection
        int time_since_update;              // Number of consecutive frames
        std::vector<double> size;           // 3D dimensions of the object bounding box [length, width, height]

        // Kalman Filter matrices
        Eigen::VectorXd x;                  // State: [x, y, z, vx, vy, vz]
        Eigen::MatrixXd P;                  // State covariance
        Eigen::MatrixXd F;                  // State transition matrix
        Eigen::MatrixXd H;                  // Measurement matrix
        Eigen::MatrixXd R;                  // Measurement noise covariance
        Eigen::MatrixXd Q;                  // Process noise covariance
        Eigen::MatrixXd I;                  // Identity matrix

        Kalman_Track(const Eigen::Vector3d& detection, const std::vector<double>& dimensions, int id) 
            : track_id(id), frames_cnt(1), match_cnt(1), time_since_update(0), size(dimensions) 
        {
            x = Eigen::VectorXd::Zero(6);                   // Initialize 6D state vector to zero, then seed the first 3 components [x, y, z] with sensor data
            x.head(3) = detection;                          // [vx, vy, vz] velocities default to 0.0 initially

            P = Eigen::MatrixXd::Identity(6, 6) * 10.0;     // Initialize state uncertainty high (10.0) because velocity is unknown on the very first frame
            
            F = Eigen::MatrixXd::Identity(6, 6);            // Set up base transition framework (identity layout; time deltas are injected during prediction)
            
            H = Eigen::MatrixXd::Zero(3, 6);                // Define extraction mapping: Sensor only measures position [x, y, z], it cannot directly see velocity
            H(0, 0) = 1.0; 
            H(1, 1) = 1.0; 
            H(2, 2) = 1.0;

            R = Eigen::MatrixXd::Identity(3, 3) * 0.5;      // Define trusted sensor accuracy (0.5 means low noise, high trust in LiDAR coordinates)
            Q = Eigen::MatrixXd::Identity(6, 6) * 0.1;      // Define internal process noise (0.1 means we expect relatively smooth, predictable movement)
            I = Eigen::MatrixXd::Identity(6, 6);
        }

        // Kalman Filter Prediction Step: Projects the object's position forward into the future using physics
        Eigen::Vector3d predict(double dt) {
            // Update physics model with actual dt
            // Dynamically inject the elapsed time (dt) into the constant-velocity kinematic equations:
            // x_new = x + vx * dt  |  y_new = y + vy * dt  |  z_new = z + vz * dt
            F(0, 3) = dt;
            F(1, 4) = dt;
            F(2, 5) = dt;

            // State Prediction: Multiplies physics matrix by current state to get new expected position
            x = F * x;
            // Uncertainty Propagation: Extrapolates covariance forward and injects process noise (uncertainty grows)
            P = F * P * F.transpose() + Q;

            // Increment life tracking metrics
            frames_cnt += 1;
            time_since_update += 1; // Assumes a miss until a successful update() occurs in the same frame
            return x.head(3); // Return only the predicted 3D position [x, y, z] for matching routines
        }

        // Kalman Filter Correction/Update Step: Corrects math predictions using fresh physical sensor data
        void update(const Eigen::Vector3d& detection, const std::vector<double>& dimensions) {
            Eigen::Vector3d y = detection - (H * x);                            // Innovation/Residual (y): The vector difference between actual sensor detection and predicted position
            Eigen::MatrixXd S = H * P * H.transpose() + R;                      // Innovation Covariance (S): Combines model uncertainty (P) and sensor noise (R)
            Eigen::MatrixXd K = P * H.transpose() * S.inverse();                // Kalman Gain (K): Determines weight—do we trust our physics model prediction or the sensor hardware more?

            x = x + (K * y);                                                    // Corrected State Estimate: Blends prediction with reality based on Kalman Gain weight factor
            P = (I - K * H) * P;                                                // Joseph Form / Standard Covariance Update: Decreases uncertainty (P) because a real sensor confirmation arrived

            size = dimensions;                                                  // Keep physical properties fresh and reset coasting countdown timers
            time_since_update = 0;                                              // Reset counter since object was successfully seen
            match_cnt += 1;                                                     // Increment confidence score
        }
};

class ObjectTrackingNode : public rclcpp::Node {
    public:
        // init next_track_id_ = 0
        // last_timestamp_ = -1.0
        // only init with the consturtor run
        ObjectTrackingNode() : Node("lidar_tracker_node"), next_track_id_(0), last_timestamp_(-1.0) 
        {
            this->declare_parameter<int>("max_missed_frames_time", 3);                      // declares a ROS 2 parameter named "max_missed_frames_time" with a default value of 3
            this->get_parameter("max_missed_frames_time", max_missed_frames_time_);         // put parameter into max_missed_frames_time_

            this->declare_parameter<double>("distance_threshold", 4.0);                     // sets the maximum Euclidean distance allowed between the Kalman filter prediction and the Lidar measurement
                                                                                            // for faster moving object should set larger value
            this->get_parameter("distance_threshold", distance_threshold_);                   

            // Create a subscription to the cluster bounding marker array
            subscription_cluster_bounding_boxes_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
                "/lidar_cluster/bounding_boxes", 10,
                std::bind(&ObjectTrackingNode::tracker_callback, this, std::placeholders::_1));

            // Create publishers for tracked object marker array
            publisher_tracked_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/lidar_track/tracked_objects", 10);

            RCLCPP_INFO(this->get_logger(), "Lidar Tracker Node Started...");
        }

    private:
        std::vector<Kalman_Track> tracks_;
        int next_track_id_;
        double last_timestamp_;

        // Tuned Parameters
        int max_missed_frames_time_;
        double distance_threshold_;

        void tracker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) 
        {
            // If the incoming ROS 2 message contains no markers (empty frame), exit early.
            if (msg->markers.empty()) 
                return;

            // Initialize tracking variables for the current time frame
            double current_time_float = 0.0;
            std::string current_frame = "map";
            builtin_interfaces::msg::Time stamp;
            bool stamp_found = false;

            // Extract raw detections & timestamp
            std::vector<Eigen::Vector3d> detections;
            std::vector<std::vector<double>> dimensions;

            // Loop through all incoming markers to extract object positions and sizes.
            for (const auto& marker : msg->markers) 
            {
                // Only process markers explicitly flagged to be added/rendered (ADD action)
                if (marker.action == visualization_msgs::msg::Marker::ADD)
                {
                    // Extract the timestamp and frame ID from the very first valid marker found, speed up the performance to avoid do * float point
                    if (!stamp_found) 
                    {
                        stamp = marker.header.stamp;
                        current_frame = marker.header.frame_id;
                        current_time_float = stamp.sec + (stamp.nanosec * 1e-9);
                        stamp_found = true;
                    }
                    detections.push_back(Eigen::Vector3d(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z));  // Parse and store the 3D position [x, y, z] using Eigen Vectors for matrix math
                    dimensions.push_back({marker.scale.x, marker.scale.y, marker.scale.z});                                         // Parse and store object dimensions [length, width, height]
                }
            }

            // Guard clause: If no valid active markers were found, terminate callback
            if (!stamp_found) 
                return;

            // DYNAMIC TIME DELTA (dt) CALCULATIONs
            // Calculate exact elapsed time since the last frame to feed the physics model.
            double dt = 0.1;                                    // Default fallback to 10Hz if it's the very first frame, define in KITTI
            if (last_timestamp_ > 0.0) 
            {
                dt = current_time_float - last_timestamp_;
                if (dt <= 0.0 || dt > 2.0)                      // Fallback constraint to prevent massive time jumps due to bag loops or rosbag pauses
                {
                    dt = 0.1; 
                }
            }
            last_timestamp_ = current_time_float;               // Update history for the next frame

            // KALMAN FILTER PREDICTION STEP
            std::vector<Eigen::Vector3d> predictions;           // Vector to store where the system expects existing objects to be right now
            for (auto& track : tracks_) 
            {
                predictions.push_back(track.predict(dt));       // Project the track's state forward by 'dt' seconds based on its current velocity
            }

            // Vectors to handle data mapping and track status categorization
            std::vector<std::pair<int, int>> matched_indices;
            std::vector<int> unmatched_detections;
            std::vector<int> unmatched_tracks;

            //  DATA ASSOCIATION (GREEDY NEAREST-NEIGHBOR)
            if (!tracks_.empty() && !detections.empty())        // Only attempt matching if we have both active tracks and new raw detections
            {
                struct MatchCost                                // Local structure to record Euclidean distances between tracks and detections
                { 
                    int track_idx;                              // index of an existing active track, historical object the system is currently managing
                    int detection_idx;                          // index of new 3D bounding box inside the current frame, raw object the Lidar sensor just detected
                    double cost;                                // 3D Euclidean distance between that specific track prediction and the new detection
                };

                std::vector<MatchCost> costs;

                // Compute distances between EVERY existing track prediction and EVERY new detection
                for (size_t t = 0; t < predictions.size(); ++t)                 // existing track prediction
                {
                    for (size_t d = 0; d < detections.size(); ++d)              // EVERY new detection
                    {
                        double dist = (predictions[t] - detections[d]).norm();  // 3D Euclidean distance
                        // Gating: Only consider this pairing if they are physically close enough
                        if (dist < distance_threshold_)
                        {
                            costs.push_back({(int)t, (int)d,dist});
                        }
                    }
                }

                // Sort the entire pairing list so the absolute closest pairs are at the front
                std::sort(costs.begin(), costs.end(), [](const MatchCost& a, const MatchCost& b) {
                    return a.cost < b.cost;
                });

                // Dynamic lookup tables to prevent a track or detection from being matched twice
                std::vector<bool> tracks_assigned(tracks_.size(), false);
                std::vector<bool> detections_assigned(detections.size(), false);

                // Greedy matching pass: Lock in the best pairs first
                for (const auto& c : costs) 
                {
                    if (!tracks_assigned[c.track_idx] && !detections_assigned[c.detection_idx]) 
                    {
                        matched_indices.push_back({c.track_idx, c.detection_idx});  // Confirm match
                        tracks_assigned[c.track_idx] = true;
                        detections_assigned[c.detection_idx] = true;
                    }
                }

                // Collect all tracks that did NOT find a corresponding sensor detection
                for (size_t t = 0; t < tracks_.size(); ++t) 
                {
                    if (!tracks_assigned[t]) 
                    {
                        unmatched_tracks.push_back(t);
                    }
                }

                // Collect all new sensor detections that did NOT match any historical track
                for (size_t d = 0; d < detections.size(); ++d)
                {
                    if (!detections_assigned[d]) 
                    {
                        unmatched_detections.push_back(d);
                    }
                }
            } 
            else 
            {
                // If no tracks exist, all detections are unmatched. 
                // If no detections exist, all tracks are unmatched.
                for (size_t d = 0; d < detections.size(); ++d) 
                {
                    unmatched_detections.push_back(d);
                }

                for (size_t t = 0; t < tracks_.size(); ++t)
                {
                    unmatched_tracks.push_back(t);
                } 
            }

            // KALMAN FILTER MEASUREMENT UPDATE STEP
            // For successfully paired objects, feed new sensor data into their Kalman Filter
            for (const auto& match : matched_indices) 
            {
                tracks_[match.first].update(detections[match.second], dimensions[match.second]);
            }

            // TRACK MANAGEMENT: BIRTH & DEATH
            // Turn every completely new unmatched detection into a freshly tracked object
            for (int detection_idx : unmatched_detections) 
            {
                tracks_.push_back(Kalman_Track(detections[detection_idx], dimensions[detection_idx], next_track_id_++));
            }

            // Wipe out old tracks that have gone undetected for too long (max_missed_frames_time)
            tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                [this](const Kalman_Track& t) 
                { 
                    return t.time_since_update > max_missed_frames_time_; 
                }), tracks_.end());

            // VISUALIZATION OUTPUT
            publish_markers(current_frame, stamp);
        }

        void publish_markers(const std::string& frame_id, const builtin_interfaces::msg::Time& stamp) {
            visualization_msgs::msg::MarkerArray marker_array;

            visualization_msgs::msg::Marker delete_all;
            delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(delete_all);

            for (const auto& track : tracks_) {
                if (track.match_cnt < 3) continue;

                double x = track.x(0), y = track.x(1), z = track.x(2);

                // Bounding Box
                visualization_msgs::msg::Marker box_marker;
                box_marker.header.frame_id = frame_id;
                box_marker.header.stamp = stamp;
                box_marker.ns = "tracked_boxes";
                box_marker.id = track.track_id * 2;
                box_marker.type = visualization_msgs::msg::Marker::CUBE;
                box_marker.action = visualization_msgs::msg::Marker::ADD;
                box_marker.pose.position.x = x; box_marker.pose.position.y = y; box_marker.pose.position.z = z;
                box_marker.scale.x = track.size[0]; box_marker.scale.y = track.size[1]; box_marker.scale.z = track.size[2];
                box_marker.color.r = 0.0; box_marker.color.g = 1.0; box_marker.color.b = 0.0; box_marker.color.a = 0.5;
                marker_array.markers.push_back(box_marker);

                // Text
                visualization_msgs::msg::Marker text_marker;
                text_marker.header = box_marker.header;
                text_marker.ns = "tracked_text";
                text_marker.id = (track.track_id * 2) + 1;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.pose.position.x = x; text_marker.pose.position.y = y; 
                text_marker.pose.position.z = z + (track.size[2] / 2.0) + 0.5;
                text_marker.scale.z = 0.6;
                text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;
                
                char text_buffer[50];
                snprintf(text_buffer, sizeof(text_buffer), "ID: %d", track.track_id);
                text_marker.text = text_buffer;
                
                marker_array.markers.push_back(text_marker);
            }

            publisher_tracked_->publish(marker_array);
        }

        rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr subscription_cluster_bounding_boxes_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_tracked_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);                                  // Initialize ROS 2
    rclcpp::spin(std::make_shared<ObjectTrackingNode>());      // Spin the node to process callback
    rclcpp::shutdown();
    return 0;
}