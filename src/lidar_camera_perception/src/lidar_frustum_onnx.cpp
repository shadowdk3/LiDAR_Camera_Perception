#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>

#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

class FrustumONNXNode : public rclcpp::Node 
{
    public:
        FrustumONNXNode() : Node("frustum_onnx_node"), frame_count_(0)
        {
            // Resolve paths
            fs::path script_dir = fs::path(__FILE__).parent_path();
            fs::path ws_dir = script_dir.parent_path().parent_path().parent_path();
            RCLCPP_INFO(this->get_logger(), "%s %s...", script_dir.c_str(), ws_dir.c_str());

            std::string model_path = (ws_dir / "src" / "lidar_camera_perception" / "frustum_pointnet" / "checkpoints" / "frustum_pointnet_fine_tune.onnx").string();
            std::string yolo_onnx_path = (ws_dir / "models" / "yolo11n.onnx").string();

            RCLCPP_INFO(this->get_logger(), "Loading ONNX model from %s...", model_path.c_str());

            // Initialize ONNX Runtime environment for Frustum PointNet
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FrustumPointNet");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            RCLCPP_INFO(this->get_logger(), "Using CPU execution provider for ONNX Runtime.");

            ort_session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

            // Setup input/output names for Frustum PointNet as std::string
            Ort::AllocatorWithDefaultOptions allocator;
            auto in_ptr = ort_session_->GetInputNameAllocated(0, allocator);
            fr_input_name_ = std::string(in_ptr.get());

            auto out_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
            fr_output_name_ = std::string(out_ptr.get());

            // Load YOLO ONNX Session
            RCLCPP_INFO(this->get_logger(), "Loading YOLO model from %s...", yolo_onnx_path.c_str());
            
            yolo_ort_session_ = std::make_unique<Ort::Session>(*env_, yolo_onnx_path.c_str(), session_options);
            
            auto yolo_in_ptr = yolo_ort_session_->GetInputNameAllocated(0, allocator);
            yolo_input_name_ = std::string(yolo_in_ptr.get());

            auto yolo_out_ptr = yolo_ort_session_->GetOutputNameAllocated(0, allocator);
            yolo_output_name_ = std::string(yolo_out_ptr.get());

            // ROS 2 Subscribers & Publishers
            pc_sub_.subscribe(this, "/kitti/point_cloud");
            img_sub_.subscribe(this, "/kitti/image/gray/left");

            sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), pc_sub_, img_sub_);
            sync_->registerCallback(std::bind(&FrustumONNXNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

            marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/frustum_bounding_boxes", 10);
            debug_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/frustum_debug_point_cloud", 10);
            yolo_debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/frustum_debug_yolo", 10);

            RCLCPP_INFO(this->get_logger(), "Frustum PointNet C++ Node initialized and listening...");
        }

    private:
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>;

        Eigen::MatrixXf convertCloud2ToMatrix(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) 
        {
            size_t point_step = msg->point_step;
            size_t num_points = msg->data.size() / point_step;
            
            Eigen::MatrixXf points(num_points, 4);
            
            for (size_t i = 0; i < num_points; ++i) 
            {
                const float* ptr = reinterpret_cast<const float*>(&msg->data[i * point_step]);
                points(i, 0) = ptr[0];
                points(i, 1) = ptr[1];
                points(i, 2) = ptr[2];
                points(i, 3) = ptr[3];
            }
            return points;
        }

        cv::Mat convertImgToMat(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
        {
            cv::Mat img;
            if (msg->encoding == "rgb8") 
            {
                cv::Mat rgb(msg->height, msg->width, CV_8UC3, const_cast<unsigned char*>(msg->data.data()), msg->step);
                cv::cvtColor(rgb, img, cv::COLOR_RGB2BGR);
            } 
            else if (msg->encoding == "bgr8") 
            {
                img = cv::Mat(msg->height, msg->width, CV_8UC3, const_cast<unsigned char*>(msg->data.data()), msg->step).clone();
            } 
            else 
            {
                img = cv::Mat(msg->height, msg->width, CV_8UC1, const_cast<unsigned char*>(msg->data.data()), msg->step).clone();
            }
            return img;
        }

        std::vector<cv::Rect> runYoloInference(cv::Mat &current_img, Ort::Session &yolo_session, 
                                   const std::string &input_name, const std::string &output_name) 
        {
            int img_w = current_img.cols;
            int img_h = current_img.rows;
            int max_dim = std::max(img_w, img_h);
            
            cv::Mat resized_img;
            cv::copyMakeBorder(current_img, resized_img, 0, max_dim - img_h, 0, max_dim - img_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            
            cv::Mat blob;
            cv::dnn::blobFromImage(resized_img, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

            size_t input_tensor_size = 1 * 3 * 640 * 640;
            std::vector<int64_t> input_shape = {1, 3, 640, 640};

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, blob.ptr<float>(), input_tensor_size, input_shape.data(), input_shape.size()
            );

            const char* input_names[] = {input_name.c_str()};
            const char* output_names[] = {output_name.c_str()};

            auto output_tensors = yolo_session.Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
            );

            float* raw_output = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            int num_channels = output_shape[1];
            int num_anchors = output_shape[2]; 

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            float scale = static_cast<float>(max_dim) / 640.0f;

            for (int i = 0; i < num_anchors; ++i) 
            {
                float cx = raw_output[0 * num_anchors + i];
                float cy = raw_output[1 * num_anchors + i];
                float w  = raw_output[2 * num_anchors + i];
                float h  = raw_output[3 * num_anchors + i];

                float max_class_score = 0.0f;
                for (int c = 4; c < num_channels; ++c) 
                {
                    float score = raw_output[c * num_anchors + i];
                    if (score > max_class_score) 
                    {
                        max_class_score = score;
                    }
                }

                if (max_class_score > 0.35f) 
                {
                    float x1 = (cx - w / 2.0f) * scale;
                    float y1 = (cy - h / 2.0f) * scale;
                    float x2 = (cx + w / 2.0f) * scale;
                    float y2 = (cy + h / 2.0f) * scale;

                    boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
                    confidences.push_back(max_class_score);
                }
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.35f, 0.45f, indices);

            std::vector<cv::Rect> final_boxes;
            for (int idx : indices) {
                final_boxes.push_back(boxes[idx]);
            }

            return final_boxes;
        }

        Eigen::Matrix<double, 3, 4> read_kitti_calib() {
            Eigen::Matrix4d velo_to_cam0_extrinsic;
            velo_to_cam0_extrinsic << 
                 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
                 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02,
                 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00;

            Eigen::Matrix4d cam0_rectification;
            cam0_rectification << 
                 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00,
                -9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00,
                 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00;

            Eigen::Matrix<double, 3, 4> cam2_projection_rectified;
            cam2_projection_rectified << 
                7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
                0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03;

            // Result is a 3x4 projection matrix mapping Velodyne directly to Cam2 image plane
            return cam2_projection_rectified * cam0_rectification * velo_to_cam0_extrinsic;
        }

        std::pair<Eigen::MatrixXf, Eigen::Vector3f> extract_frustum_data_from_pc(
            const Eigen::MatrixXf &point_cloud, const cv::Rect &box2d) 
        {
            Eigen::Matrix<double, 3, 4> velo_cam2_projection = read_kitti_calib();
            
            int num_pts = point_cloud.rows();
            Eigen::MatrixXf pts_3d = point_cloud.leftCols(3);
            
            Eigen::Matrix4d velo_to_cam0_ext;
            velo_to_cam0_ext << 
                 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
                 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02,
                 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00;

            std::vector<int> valid_indices;
            std::vector<Eigen::Vector3f> frustum_pts_vec;
            std::vector<Eigen::Vector3f> pts_cam_in_box_vec;

            for (int i = 0; i < num_pts; ++i) {
                Eigen::Vector4d pt_h(pts_3d(i, 0), pts_3d(i, 1), pts_3d(i, 2), 1.0);
                
                // Get 3D point in camera frame to check depth (z > 0.1)
                Eigen::Vector4d pt_cam_h = velo_to_cam0_ext * pt_h;
                if (pt_cam_h[2] <= 0.1) continue;

                // Project to 2D image coordinates
                Eigen::Vector3d pt_2d_homo = velo_cam2_projection * pt_h;
                
                double u = pt_2d_homo[0] / pt_2d_homo[2];
                double v = pt_2d_homo[1] / pt_2d_homo[2];

                if (u >= box2d.x && u <= (box2d.x + box2d.width) &&
                    v >= box2d.y && v <= (box2d.y + box2d.height)) 
                {
                    frustum_pts_vec.push_back(pts_3d.row(i));
                    pts_cam_in_box_vec.push_back(pt_cam_h.head<3>().cast<float>());
                }
            }

            if (frustum_pts_vec.size() < 5) {
                return {Eigen::MatrixXf(), Eigen::Vector3f::Zero()};
            }

            // Height filtering based on camera frame Y percentiles
            if (pts_cam_in_box_vec.size() > 30) {
                std::vector<float> y_vals;
                for (const auto &pt : pts_cam_in_box_vec) y_vals.push_back(pt[1]);
                
                size_t p5_idx = static_cast<size_t>(0.05 * y_vals.size());
                size_t p90_idx = static_cast<size_t>(0.90 * y_vals.size());
                std::sort(y_vals.begin(), y_vals.end());
                float y_min = y_vals[p5_idx];
                float y_max = y_vals[p90_idx];

                std::vector<Eigen::Vector3f> filtered_pts;
                for (size_t i = 0; i < frustum_pts_vec.size(); ++i) {
                    if (pts_cam_in_box_vec[i][1] >= y_min && pts_cam_in_box_vec[i][1] <= y_max) {
                        filtered_pts.push_back(frustum_pts_vec[i]);
                    }
                }
                frustum_pts_vec = filtered_pts;
            }

            if (frustum_pts_vec.size() < 5) {
                return {Eigen::MatrixXf(), Eigen::Vector3f::Zero()};
            }

            Eigen::MatrixXf frustum_pts(frustum_pts_vec.size(), 3);
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            for (size_t i = 0; i < frustum_pts_vec.size(); ++i) {
                frustum_pts.row(i) = frustum_pts_vec[i];
                centroid += frustum_pts_vec[i];
            }
            centroid /= static_cast<float>(frustum_pts_vec.size());

            return {frustum_pts, centroid};
        }

        Eigen::MatrixXf sample_and_normalize_points(const Eigen::MatrixXf &frustum_pts, int num_samples = 512) {
            Eigen::Vector3f centroid = frustum_pts.colwise().mean();
            Eigen::MatrixXf norm_pts = frustum_pts.rowwise() - centroid.transpose();

            int num_pts = norm_pts.rows();
            Eigen::MatrixXf sampled_pts(num_samples, 3);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, num_pts - 1);

            for (int i = 0; i < num_samples; ++i) {
                int idx = dis(gen);
                sampled_pts.row(i) = norm_pts.row(idx);
            }
            return sampled_pts;
        }

        std::tuple<float, float, float, float> yaw_to_quaternion(float yaw) {
            float qx = 0.0f;
            float qy = 0.0f;
            float qz = std::sin(yaw * 0.5f);
            float qw = std::cos(yaw * 0.5f);
            return {qx, qy, qz, qw};
        }

        void publish_marker_array(const std::vector<Eigen::VectorXf> &pred_boxes, const std_msgs::msg::Header &header) {
            visualization_msgs::msg::MarkerArray marker_array;

            visualization_msgs::msg::Marker del_marker;
            del_marker.header = header;
            del_marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(del_marker);

            for (size_t i = 0; i < pred_boxes.size(); ++i) {
                float x = pred_boxes[i][0];
                float y = pred_boxes[i][1];
                float z = pred_boxes[i][2];
                float l = pred_boxes[i][3];
                float w = pred_boxes[i][4];
                float h = pred_boxes[i][5];
                float rz = pred_boxes[i][6];

                visualization_msgs::msg::Marker marker;
                marker.header = header;
                marker.ns = "frustum_3d_boxes";
                marker.id = i;
                marker.type = visualization_msgs::msg::Marker::CUBE;
                marker.action = visualization_msgs::msg::Marker::ADD;

                marker.pose.position.x = static_cast<double>(x);
                marker.pose.position.y = static_cast<double>(y);
                marker.pose.position.z = static_cast<double>(z);

                auto [qx, qy, qz, qw] = yaw_to_quaternion(rz);
                marker.pose.orientation.x = qx;
                marker.pose.orientation.y = qy;
                marker.pose.orientation.z = qz;
                marker.pose.orientation.w = qw;

                marker.scale.x = static_cast<double>(l);
                marker.scale.y = static_cast<double>(w);
                marker.scale.z = static_cast<double>(h);

                marker.color.r = 1.0f;
                marker.color.g = 0.0f;
                marker.color.b = 0.0f;
                marker.color.a = 0.6f;

                marker.lifetime.sec = 0;
                marker.lifetime.nanosec = 200000000;

                marker_array.markers.push_back(marker);
            }
            marker_pub_->publish(marker_array);
        }

        void publish_debug_point_cloud(const Eigen::MatrixXf &points_np, const std_msgs::msg::Header &header) {
            sensor_msgs::msg::PointCloud2 debug_msg;
            debug_msg.header = header;
            debug_msg.height = 1;
            debug_msg.width = points_np.rows();
            debug_msg.fields.resize(3);

            debug_msg.fields[0].name = "x"; debug_msg.fields[0].offset = 0; debug_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; debug_msg.fields[0].count = 1;
            debug_msg.fields[1].name = "y"; debug_msg.fields[1].offset = 4; debug_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; debug_msg.fields[1].count = 1;
            debug_msg.fields[2].name = "z"; debug_msg.fields[2].offset = 8; debug_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; debug_msg.fields[2].count = 1;

            debug_msg.point_step = 12;
            debug_msg.row_step = debug_msg.point_step * debug_msg.width;
            debug_msg.data.resize(debug_msg.row_step);

            unsigned char *ptr = debug_msg.data.data();
            for (int i = 0; i < points_np.rows(); ++i) {
                float x = points_np(i, 0);
                float y = points_np(i, 1);
                float z = points_np(i, 2);
                std::memcpy(ptr + 0, &x, sizeof(float));
                std::memcpy(ptr + 4, &y, sizeof(float));
                std::memcpy(ptr + 8, &z, sizeof(float));
                ptr += debug_msg.point_step;
            }
            debug_pc_pub_->publish(debug_msg);
        }

        void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc_msg, const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
            frame_count_++;
            RCLCPP_INFO(this->get_logger(), "Processing synchronized frame %d...", frame_count_);

            Eigen::MatrixXf current_pc = convertCloud2ToMatrix(pc_msg);
            cv::Mat current_img = convertImgToMat(img_msg);

            std::vector<cv::Rect> boxes = runYoloInference(
                current_img, 
                *yolo_ort_session_, 
                yolo_input_name_, 
                yolo_output_name_ 
            );

            cv::Mat debug_img = current_img.clone();

            std::vector<Eigen::MatrixXf> all_frustum_pts;
            std::vector<Eigen::VectorXf> pred_boxes_list;

            for (const auto &box : boxes) {
                int debug_u1 = box.x;
                int debug_v1 = box.y;
                int debug_u2 = box.x + box.width;
                int debug_v2 = box.y + box.height;

                auto [frustum_pts, centroid] = extract_frustum_data_from_pc(current_pc, box);
                if (frustum_pts.rows() < 5) 
                {
                    continue;
                }

                all_frustum_pts.push_back(frustum_pts);
                Eigen::MatrixXf sampled_pts = sample_and_normalize_points(frustum_pts, 512);

                // Prepare input tensor for Frustum PointNet (shape: 1 x 512 x 3)
                std::vector<int64_t> input_shape = {1, 512, 3};
                size_t input_tensor_size = 512 * 3;

                std::vector<float> input_tensor_values(input_tensor_size);
                int flat_idx = 0;
                for (int i = 0; i < sampled_pts.rows(); ++i) 
                {
                    for (int j = 0; j < sampled_pts.cols(); ++j) 
                    {
                        input_tensor_values[flat_idx++] = sampled_pts(i, j);
                    }
                }

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_size, 
                    input_shape.data(), input_shape.size()
                );

                const char* input_names[] = {fr_input_name_.c_str()};
                const char* output_names[] = {fr_output_name_.c_str()};

                auto output_tensors = ort_session_->Run(
                    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
                );

                float* raw_output = output_tensors[0].GetTensorMutableData<float>();
                auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                int output_dim = output_shape[1];

                Eigen::VectorXf pred_box(output_dim);
                for (int i = 0; i < output_dim; ++i) 
                {
                    pred_box[i] = raw_output[i];
                }

                // Add centroid offset to position [x, y, z]
                if (output_dim >= 3) 
                {
                    pred_box[0] += centroid[0];
                    pred_box[1] += centroid[1];
                    pred_box[2] += centroid[2];
                }

                pred_boxes_list.push_back(pred_box);
                cv::rectangle(debug_img, cv::Point(debug_u1, debug_v1), cv::Point(debug_u2, debug_v2), cv::Scalar(0, 255, 0), 2);
            }

            publish_marker_array(pred_boxes_list, pc_msg->header);
            
            // Publish combined frustum points to visually verify what the network "saw"
            if (!all_frustum_pts.empty()) 
            {
                int total_rows = 0;
                for (const auto &mat : all_frustum_pts) 
                {
                    total_rows += mat.rows();
                }

                Eigen::MatrixXf combined_pts(total_rows, 3);
                int current_row = 0;
                for (const auto &mat : all_frustum_pts) 
                {
                    combined_pts.block(current_row, 0, mat.rows(), 3) = mat;
                    current_row += mat.rows();
                }

                publish_debug_point_cloud(combined_pts, pc_msg->header);
            }

            sensor_msgs::msg::Image::SharedPtr debug_img_msg = cv_bridge::CvImage(pc_msg->header, "bgr8", debug_img).toImageMsg();
            yolo_debug_pub_->publish(*debug_img_msg);
        }

        // Member variables
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> ort_session_;
        std::unique_ptr<Ort::Session> yolo_ort_session_;
        
        Ort::AllocatorWithDefaultOptions allocator_;

        std::string fr_input_name_;
        std::string fr_output_name_;
        std::string yolo_input_name_;
        std::string yolo_output_name_;

        message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pc_sub_;
        message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pc_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr yolo_debug_pub_;

        int frame_count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FrustumONNXNode>());
    rclcpp::shutdown();
    return 0;
}