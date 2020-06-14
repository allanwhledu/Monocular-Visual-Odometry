// The member function `addFrame` of class VisualOdometry is defined here.

#include "my_slam/vo/vo.h"

namespace my_slam
{
namespace vo
{

void VisualOdometry::addFrame(Frame::Ptr frame)
{
    // Settings
    pushFrameToBuff_(frame);

    // Renamed vars
    curr_ = frame;
    const int img_id = curr_->id_;
    const cv::Mat &K = curr_->camera_->K_;

    // Start
    printf("\n\n=============================================\n");
    printf("Start processing the %dth image.\n", img_id);

    curr_->calcKeyPoints();
    curr_->calcDescriptors();
    cout << "Number of keypoints: " << curr_->keypoints_.size() << endl;
    prev_ref_ = ref_;

    // vo_state_: BLANK -> DOING_INITIALIZATION
    if (vo_state_ == BLANK)
    {
        curr_->T_w_c_ = cv::Mat::eye(4, 4, CV_64F);
        vo_state_ = DOING_INITIALIZATION;
        addKeyFrame_(curr_); // curr_ becomes the ref_
    }
    else if (vo_state_ == DOING_INITIALIZATION)
    {
        // Match features
        geometry::matchFeatures(ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_);
        printf("Number of matches with the 1st frame: %d\n", (int)curr_->matches_with_ref_.size());

        // Estimae motion and triangulate points
//        这里应该是估计相机运动，并且获得匹配点的3D坐标
//应该是初始化的时候才需要用三角测量，之后tracking的时候都是用pnp了是吧？
        estimateMotionAnd3DPoints_();
        printf("Number of inlier matches: %d\n", (int)curr_->inliers_matches_for_3d_.size());

        // Check initialization condition:
        printf("\nCheck VO init conditions: \n");
        if (isVoGoodToInit_())
        {
            cout << "Large movement detected at frame " << img_id << ". Start initialization" << endl;
            pushCurrPointsToMap_();
            addKeyFrame_(curr_);
            vo_state_ = DOING_TRACKING;
            cout << "Inilialiation success !!!" << endl;
            cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        }
        else // skip this frame
        {
            curr_->T_w_c_ = ref_->T_w_c_;
            cout << "Not initialize VO ..." << endl;
        }
    }
    else if (vo_state_ == DOING_TRACKING)
    {
        printf("\nDoing tracking\n");
        curr_->T_w_c_ = ref_->T_w_c_.clone(); // Initial estimation of the current pose 首先将当前帧的世界坐标转换定义成和上一帧相同
        bool is_pnp_good = poseEstimationPnP_(); //这里面要是实现特征点的检测、描述子的获取、特征点的匹配、3D坐标的获取，最后是PNP
        if (!is_pnp_good) // pnp failed. Print log.
        {
            int num_matches = curr_->matches_with_map_.size();
            constexpr int kMinPtsForPnP = 5;
            printf("PnP failed.\n");
            printf("    Num inlier matches: %d.\n", num_matches);
            if (num_matches >= kMinPtsForPnP)
            {
                printf("    Computed world to camera transformation:\n");
                std::cout << curr_->T_w_c_ << std::endl;
            }
            printf("PnP result has been reset as R=identity, t=zero.\n");
        }
        else // pnp good
        {
            callBundleAdjustment_(); //todo 之后在看这个ba吧
            // -- Insert a keyframe is motion is large. Then, triangulate more points
            if (check_Large_Move_For_Add_KeyFrame_(curr_, ref_)) //看看是不是关键帧 todo 关键帧要做什么？
            {
                // Feature matching 不是关键帧的时候是当前帧和路标点进行比较，如果是关键帧就要和上一帧进行比较了 todo 可以实施不要关键帧呢？是不是会一直连续进行
                geometry::matchFeatures(ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_);

                // Find inliers by epipolar constraint //求本质矩阵的时候，给出内点的序号？
                curr_->inliers_matches_with_ref_ = geometry::helper_Find_Inlier_Matches_By_EpipolarCons(
                        ref_->keypoints_, curr_->keypoints_, curr_->matches_with_ref_, K);

                // Print
                printf("For triangulation: Matches with prev keyframe: %d; Num inliers: %d \n",
                       (int)curr_->matches_with_ref_.size(), (int)curr_->inliers_matches_with_ref_.size());
                // 所以，当是关键帧的时候，就用2d-2d来进行帧间匹配的啊！！！
                // Triangulate points 求ref帧的点在curr中的位姿
                curr_->inliers_pts3d_ = geometry::helperTriangulatePoints(
                    ref_->keypoints_, curr_->keypoints_,
                    curr_->inliers_matches_with_ref_, getMotionFromFrame1to2(curr_, ref_), K);

                retain_good_triangulation_result();

                // -- Update state
                pushCurrPointsToMap_();
                optimizeMap_(); //将路标点还进行了一次丢弃
                addKeyFrame_(curr_);
            }
        }
    }

    // Print relative motion //刚才那一帧流程下来不是keyframe有点可惜
    if (vo_state_ == DOING_TRACKING)
    {
        static cv::Mat T_w_to_prev = cv::Mat::eye(4, 4, CV_64F);  //TODO 为什么上一帧是111，有问题吧？
        const cv::Mat &T_w_to_curr = curr_->T_w_c_;
        cv::Mat T_prev_to_curr = T_w_to_prev.inv() * T_w_to_curr;
        cv::Mat R, t;
        basics::getRtFromT(T_prev_to_curr, R, t);
        cout << "\nCamera motion:" << endl;
        cout << "R_prev_to_curr: " << R << endl;
        cout << "t_prev_to_curr: " << t.t() << endl;
    }
    prev_ = curr_;
    cout << "\nEnd of a frame" << endl;
}

} // namespace vo
} // namespace my_slam
