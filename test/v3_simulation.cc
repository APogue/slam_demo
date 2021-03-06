//
// Created by alexie on 1/27/21.
//

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <fstream>
#include <reprojection_error.h>
#include "transformation.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_error.h"
#include "imu_error.h"

#define _USE_MATH_DEFINES

class State {

public:
    State(double timestamp) {
        timestamp_ = timestamp;

        rotation_block_ptr_ = new QuatParameterBlock();
        velocity_block_ptr_ = new Vec3dParameterBlock();
        position_block_ptr_ = new Vec3dParameterBlock();
    }

    ~State() {
        delete [] rotation_block_ptr_;
        delete [] velocity_block_ptr_;
        delete [] position_block_ptr_;
    }

    double GetTimestamp() {
        return timestamp_;
    }

    QuatParameterBlock* GetRotationBlock() {
        return rotation_block_ptr_;
    }

    Vec3dParameterBlock* GetVelocityBlock() {
        return velocity_block_ptr_;
    }

    Vec3dParameterBlock* GetPositionBlock() {
        return position_block_ptr_;
    }

private:
    double timestamp_;
    QuatParameterBlock* rotation_block_ptr_;
    Vec3dParameterBlock* velocity_block_ptr_;
    Vec3dParameterBlock* position_block_ptr_;
};


struct IMU {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d gyro_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d accel_ = Eigen::Vector3d(0, 0, 0);

};

class Landmarks {
  public:
    Landmarks() {
      landmark_ptr_ = new Vec3dParameterBlock();
    }
    ~Landmarks() {
      delete [] landmark_ptr_;
    }
//  private:
    Vec3dParameterBlock* landmark_ptr_;

};
struct ObservationData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ObservationData(double timestamp) {
      timestamp_ = timestamp;
    }
    Eigen::Matrix2d cov() {
      double sigma_2 = 1e-7;
      return sigma_2 * Eigen::Matrix2d::Identity();
    }
    double timestamp_;
    size_t landmark_id_;
    Eigen::Vector2d feature_pos_;
};

Eigen::Quaterniond quat_pos(Eigen::Quaterniond q){
    if (q.w() < 0) {
        q.w() = (-1)*q.w();
        q.x() = (-1)*q.x();
        q.y() = (-1)*q.y();
        q.z() = (-1)*q.z();
    }
    return q;
};

void create_csv(std::vector<State*> state_vec, const std::string& file_path){
  std::ofstream output_file(file_path);
  output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

  for (auto & i : state_vec) {
    output_file << std::to_string(i->GetTimestamp()) << ",";
    output_file << std::to_string(i->GetPositionBlock()->estimate()(0)) << ",";
    output_file << std::to_string(i->GetPositionBlock()->estimate()(1)) << ",";
    output_file << std::to_string(i->GetPositionBlock()->estimate()(2)) << ",";
    output_file << std::to_string(i->GetVelocityBlock()->estimate()(0)) << ",";
    output_file << std::to_string(i->GetVelocityBlock()->estimate()(1)) << ",";
    output_file << std::to_string(i->GetVelocityBlock()->estimate()(2)) << ",";
    output_file << std::to_string(i->GetRotationBlock()->estimate().w()) << ",";
    output_file << std::to_string(i->GetRotationBlock()->estimate().x()) << ",";
    output_file << std::to_string(i->GetRotationBlock()->estimate().y()) << ",";
    output_file << std::to_string(i->GetRotationBlock()->estimate().z()) << std::endl;
  }
  output_file.close();
};

void create_lmk_csv(std::vector<Landmarks*> landmark_vec, const std::string& file_path){
  std::ofstream output_file(file_path);
  output_file << "p_x,p_y,p_z\n";

  for (size_t i=0; i<landmark_vec.size(); ++i) {
    output_file << std::to_string(landmark_vec.at(i)->landmark_ptr_->estimate()(0)) << ",";
    output_file << std::to_string(landmark_vec.at(i)->landmark_ptr_->estimate()(1)) << ",";
    output_file << std::to_string(landmark_vec.at(i)->landmark_ptr_->estimate()(2)) << std::endl;
  }
  output_file.close();
};


class SimSLAM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // parameters
  int state_length =500;
  int landmark_length = 4*125;
  double sigma_g_c = 8.0e-4;
  double sigma_a_c = 6.0e-3;
  double del_t = 0.02;
  double init_t = 0.0; // time
  double r = 5.0; // circle radius x-y plane
  double w = .76; // angular velocity
  double r_z = (1.0/20)*r;
  double w_z = (2.3)*w;
  double z_h = 0.0; // height of the uav
  double box_xy = 2;  // box offset from the circle
  double box_z = 1;   // box offset from uav height
  double du = 500.0;  // image dimension
  double dv = 1000.0;
  double fu = 500.0;  // focal length
  double fv = 500.0;
  double cu = 0.0;    // principal point
  double cv = 0.0;

  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);

  // create containers
  std::vector<State*> gt_vec_;
  std::vector<IMU*> imu_vec_;
  std::vector<State*> dr_vec_;
  std::vector<Landmarks*> landmark_vec_;
  std::vector<Landmarks*> gt_landmark_vec_;
  std::vector<Landmarks*> prot_landmark_vec_;
  std::vector<Landmarks*> nrot_landmark_vec_;
  std::vector<std::vector<ObservationData*>>  observation_vec_;
  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;
public:
  bool SetGroundTruth() {
    double T = init_t;
    for (unsigned i = 0; i < state_length; i++) {
        Eigen::Matrix<double, 3, 3> Tr;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Eigen::Quaterniond quat;

        pos(0) = r * cos(w * T);
        pos(1) = r * sin(w * T);
        pos(2) = r_z * sin(w_z * T) + z_h;

        vel(0) = -r * w * sin(w * T);
        vel(1) = r * w * cos(w * T);
        vel(2) = r_z * w_z * cos(w_z * T);

        Tr << cos(w * T), -sin(w * T), 0,
                sin(w * T), cos(w * T), 0,
                0, 0, 1;

        quat = Tr;
        quat = quat_pos(quat);

        gt_vec_.push_back(new State(T));

        gt_vec_.at(i)->GetPositionBlock()->setEstimate(pos);
        gt_vec_.at(i)->GetVelocityBlock()->setEstimate(vel);
        gt_vec_.at(i)->GetRotationBlock()->setEstimate(quat);

        T = T + del_t;
    }

    return true;
  }

  bool GetIMUData() {
  // IMU -> est. trajectory
    double T = init_t;
    for (unsigned i=0; i<state_length-1; i++) {
        IMU* new_imu_ptr = new IMU;
        Eigen::Vector3d a_N;
        Eigen::Vector3d omega_B = Eigen::Vector3d(0, 0, 0);

        a_N(0) = -r*(w*w)*cos(w*T);
        a_N(1) = -r*(w*w)*sin(w*T);
        a_N(2) = -r_z*(w_z*w_z)*sin(w_z*T);

        omega_B(2) = w;

        Eigen::Vector3d gyr_noise = sigma_g_c/sqrt(del_t)*Eigen::Vector3d::Random();
        Eigen::Vector3d acc_noise = sigma_a_c/sqrt(del_t)*Eigen::Vector3d::Random();
//        Eigen::Vector3d gyr_noise = Eigen::Vector3d(0, 0, 0);
//        Eigen::Vector3d acc_noise = Eigen::Vector3d(0, 0, 0);


        new_imu_ptr->gyro_ = omega_B + gyr_noise;
        new_imu_ptr->accel_= gt_vec_.at(i)->GetRotationBlock()->estimate().toRotationMatrix().transpose() *
                (a_N - gravity) + acc_noise; // expressed in the body frame

        imu_vec_.push_back(new_imu_ptr);
        T = T + del_t;
    }
    return true;
  }
  void set_init_traj(const Eigen::Vector3d& p0,
                     const Eigen::Vector3d& v0, const Eigen::Quaterniond& q0){
    double T = init_t;
    dr_vec_.push_back(new State(T)); //how can this be pushed back inside the function?
    dr_vec_.at(0)->GetPositionBlock()->setEstimate(p0);
    dr_vec_.at(0)->GetVelocityBlock()->setEstimate(v0);
    dr_vec_.at(0)->GetRotationBlock()->setEstimate(q0);
  }
  void add_noise_traj() {
    for (unsigned i = 1; i < state_length; i++) {
      Eigen::Vector3d p_noise_ = dr_vec_.at(i)->GetPositionBlock()->estimate() + .15*Eigen::Vector3d::Random();
      Eigen::Vector3d v_noise_ = dr_vec_.at(i)->GetVelocityBlock()->estimate() + .15*Eigen::Vector3d::Random();
      dr_vec_.at(i)->GetPositionBlock()->setEstimate(p_noise_);
      dr_vec_.at(i)->GetVelocityBlock()->setEstimate(v_noise_);
    }
  }
  bool SetDeadReckoning(){
    double T = init_t;
    // Dead reckoning -> est. trajectory
    Eigen::Vector3d p_dr = gt_vec_.at(0)->GetPositionBlock()->estimate();
    Eigen::Vector3d v_dr = gt_vec_.at(0)->GetVelocityBlock()->estimate();
    Eigen::Quaterniond q_dr = gt_vec_.at(0)->GetRotationBlock()->estimate();
    set_init_traj(p_dr, v_dr, q_dr);  //set the initial point
    for (unsigned i=1; i<state_length; i++) {
      T = T + del_t;
      p_dr = dr_vec_.at(i-1)->GetPositionBlock()->estimate() +
              del_t * dr_vec_.at(i-1)->GetVelocityBlock()->estimate() + 0.5 * (del_t * del_t)*
              (dr_vec_.at(i-1)->GetRotationBlock()->estimate().toRotationMatrix()
              *imu_vec_.at(i-1)->accel_ + gravity);

      v_dr = dr_vec_.at(i-1)->GetVelocityBlock()->estimate() + del_t *
              (dr_vec_.at(i-1)->GetRotationBlock()->estimate().toRotationMatrix()*
              imu_vec_.at(i-1)->accel_ + gravity);

      q_dr = dr_vec_.at(i-1)->GetRotationBlock()->estimate().normalized()*
              Exp(del_t * imu_vec_.at(i-1)->gyro_);
      q_dr = quat_pos(q_dr);

      dr_vec_.push_back(new State(T));
      dr_vec_.at(i)->GetPositionBlock()->setEstimate(p_dr);
      dr_vec_.at(i)->GetVelocityBlock()->setEstimate(v_dr);
      dr_vec_.at(i)->GetRotationBlock()->setEstimate(q_dr);
    }
//    add_noise_traj(); // add noise to check the optimization
    return true;
  }
  bool BuildLandmarkPoints(){
    Eigen::Vector3d lmk_pos;
    for (unsigned i=0; i< landmark_length/4; i++) { //x walls first
      landmark_vec_.push_back(new Landmarks);
      gt_landmark_vec_.push_back(new Landmarks);
      lmk_pos(0) = (r+box_xy)*Eigen::Vector3d::Random()(0);
      lmk_pos(1) = (r+box_xy);
      lmk_pos(2) = (box_z)*Eigen::Vector3d::Random()(2) + z_h;
      landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos + .05*Eigen::Vector3d::Random()); //triangularization noise
      gt_landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos);
//      std::cout<<"landmark"<<" "<< 2*gt_landmark_vec_.at(i)->landmark_ptr_->estimate()<<std::endl;
    }
    for (unsigned i=landmark_length/4; i<landmark_length/2; i++) { //x walls first
      landmark_vec_.push_back(new Landmarks);
      gt_landmark_vec_.push_back(new Landmarks);
      lmk_pos(0) = (r+box_xy)*Eigen::Vector3d::Random()(0);
      lmk_pos(1) = -(r+box_xy);
      lmk_pos(2) = (box_z)*Eigen::Vector3d::Random()(2) + z_h;
      landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos+.05*Eigen::Vector3d::Random());
      gt_landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos);
    }
    for (unsigned i=landmark_length/2; i< 3*landmark_length/4; i++) { //x walls first
      landmark_vec_.push_back(new Landmarks);
      gt_landmark_vec_.push_back(new Landmarks);
      lmk_pos(0) = (r+box_xy);
      lmk_pos(1) = (r+box_xy)*Eigen::Vector3d::Random()(1);
      lmk_pos(2) = (box_z)*Eigen::Vector3d::Random()(2) + z_h;
      landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos+.05*Eigen::Vector3d::Random());
      gt_landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos);
    }
    for (unsigned i=3*landmark_length/4; i< landmark_length; i++) { //x walls first
      landmark_vec_.push_back(new Landmarks);
      gt_landmark_vec_.push_back(new Landmarks);
      lmk_pos(0) = -(r+box_xy);
      lmk_pos(1) = (r+box_xy)*Eigen::Vector3d::Random()(1);
      lmk_pos(2) = (box_z)*Eigen::Vector3d::Random()(2) + z_h;
      landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos+.05*Eigen::Vector3d::Random());
      gt_landmark_vec_.at(i) ->landmark_ptr_->setEstimate(lmk_pos);
    }
  }
  bool CreateObservation()  {
    Eigen::Matrix4d T_bn;
    Eigen::Matrix4d T_cb;
    Eigen::Matrix4d T_nb;

    // from body frame to camera frame
    T_cb << cos(M_PI/2), 0, -sin(M_PI/2), 0,
            0,            1,  0,            0,
            sin(M_PI/2), 0,  cos(M_PI/2), 0,
            0, 0, 0, 1;
    observation_vec_.resize(state_length);
    for (unsigned i=0; i< state_length; i++) { //x walls first
      // from navigation to body frame
      T_bn.topLeftCorner<3, 3>() = gt_vec_.at(i)->GetRotationBlock()->estimate().toRotationMatrix().transpose();
      T_bn.topRightCorner<3, 1>() = -1 * gt_vec_.at(i)->GetRotationBlock()->estimate().toRotationMatrix().transpose() *
                                    gt_vec_.at(i)->GetPositionBlock()->estimate();
      T_bn.bottomLeftCorner<1, 3>().setZero();
      T_bn.bottomRightCorner<1, 1>().setOnes();


      // from body frame to navigation frame
      T_nb.topLeftCorner<3, 3>() = gt_vec_.at(i)->GetRotationBlock()->estimate().toRotationMatrix();
      T_nb.topRightCorner<3, 1>() = gt_vec_.at(i)->GetPositionBlock()->estimate();
      T_nb.bottomLeftCorner<1, 3>().setZero();
      T_nb.bottomRightCorner<1, 1>().setOnes();
      int count = 0;
      for (unsigned k = 0; k < landmark_length; k++) { //x walls first
        // homogeneous transformation of the landmark to camera frame
        Eigen::Vector4d homogenous_lmk_vec = Eigen::Vector4d(0, 0, 0, 1);
        homogenous_lmk_vec.head(3) = gt_landmark_vec_.at(k)->landmark_ptr_->estimate();
        Eigen::Vector4d landmark_c = T_cb * T_bn * homogenous_lmk_vec;
        if (landmark_c(2) > 0) {
          Eigen::Vector2d feature_pt;
          feature_pt(0) = fu * landmark_c(0) / landmark_c(2) + cu;
          feature_pt(1) = fv * landmark_c(1) / landmark_c(2) + cv;
          // check whether this point is in the frame
          if (abs(feature_pt(0)) <= du/2 && abs(feature_pt(1)) <= dv/2) {
            double T = gt_vec_.at(i)->GetTimestamp();
            ObservationData* feature_obs_ptr = new ObservationData(T);
            feature_obs_ptr->landmark_id_ = k;
            feature_obs_ptr->feature_pos_ = feature_pt+1e-7*Eigen::Vector2d::Random();
            observation_vec_.at(i).push_back(feature_obs_ptr);
//            prot_landmark_vec_.push_back(gt_landmark_vec_.at(k));
//            std::cout<<"landmarks"<<" "<< gt_landmark_vec_.at(k)->landmark_ptr_->estimate().size()<<std::endl;
            count++;
            }
//          else {
//
//            nrot_landmark_vec_.push_back(gt_landmark_vec_.at(k));
//          }

        }
//        else {
//          nrot_landmark_vec_.push_back(gt_landmark_vec_.at(k));
//        }
      }
//      std::cout<<"sub observation vec size"<< " " << observation_vec_.at(i).size()<<std::endl;
//      std::cout<<"count"<< " " << count<<std::endl;

    }
//    std::cout<<"observation vec size"<<" "<< observation_vec_.size()<<std::endl;
  }

  bool OutputTrajectoryResult(){
    create_csv(gt_vec_, "trajectory.csv");
    create_csv(dr_vec_, "trajectory_dr.csv");
    create_lmk_csv(gt_landmark_vec_, "landmarks.csv");
//    create_lmk_csv(prot_landmark_vec_, "prot_landmarks.csv");
//    create_lmk_csv(nrot_landmark_vec_, "nrot_landmarks.csv");
    return true;
  }

  bool SetupOptProblem() {
    Eigen::Matrix4d T_cb;
    T_cb << cos(M_PI/2), 0, -sin(M_PI/2), 0,
            0,            1,  0,            0,
            sin(M_PI/2), 0,  cos(M_PI/2), 0,
            0, 0, 0, 1;
    for (size_t i=0; i<dr_vec_.size(); ++i) {
      optimization_problem_.AddParameterBlock(dr_vec_.at(i)->GetRotationBlock()->parameters(), 4);
      optimization_problem_.AddParameterBlock(dr_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      optimization_problem_.AddParameterBlock(dr_vec_.at(i)->GetPositionBlock()->parameters(), 3);
    }
    for (size_t i=0; i<landmark_vec_.size(); ++i) {
      optimization_problem_.AddParameterBlock(landmark_vec_.at(i)->landmark_ptr_->parameters(), 3);
    }

    // imu constraints
    for (size_t i=0; i<imu_vec_.size(); ++i) {
      ceres::CostFunction* cost_function = new ImuError(imu_vec_.at(i)->gyro_,
                                                        imu_vec_.at(i)->accel_,
                                                        del_t,
                                                        Eigen::Vector3d(0,0,0),
                                                        Eigen::Vector3d(0,0,0),
                                                        sigma_g_c,
                                                        sigma_a_c);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             dr_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                             dr_vec_.at(i+1)->GetVelocityBlock()->parameters(),
                                             dr_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                             dr_vec_.at(i)->GetRotationBlock()->parameters(),
                                             dr_vec_.at(i)->GetVelocityBlock()->parameters(),
                                             dr_vec_.at(i)->GetPositionBlock()->parameters());
    }
    // observation constraints
    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_;
//        std::cout<<"ID"<<" "<<observation_vec_.at(i).at(j)->landmark_id_<<std::endl;
//        std::cout<<"Pos UV"<<" "<<observation_vec_.at(i).at(j)->feature_pos_<<std::endl;
//        std::cout<<"POS XYZ"<<" "<<landmark_vec_.at(landmark_idx)->landmark_ptr_->estimate()<<std::endl;
        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_cb.transpose(),
                                                                   fu, fv,
                                                                   cu, cv,
                                                                   observation_vec_.at(i).at(j)->cov());

        optimization_problem_.AddResidualBlock(cost_function,
                                               NULL, //loss_function_ptr_,
                                               dr_vec_.at(i)->GetRotationBlock()->parameters(),
                                               dr_vec_.at(i)->GetPositionBlock()->parameters(),
                                               landmark_vec_.at(landmark_idx)->landmark_ptr_->parameters());
      }
    }
      optimization_problem_.SetParameterBlockConstant(dr_vec_.at(0)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(dr_vec_.at(0)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(dr_vec_.at(0)->GetPositionBlock()->parameters());
      return true;
      }

  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-20;
    optimization_options_.parameter_tolerance = 1e-25;
    optimization_options_.max_num_iterations = 100;


    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;

  }

  bool OutputOptResult(){
    create_csv(dr_vec_, "trajectory_opt.csv");
    return true;
  }
};

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  SimSLAM traj_problem;

  traj_problem.SetGroundTruth();
  traj_problem.GetIMUData();
  traj_problem.SetDeadReckoning();
  traj_problem.BuildLandmarkPoints();
  traj_problem.CreateObservation();
  traj_problem.OutputTrajectoryResult();
  traj_problem.SetupOptProblem();
  traj_problem.SolveOptimizationProblem();
  traj_problem.OutputOptResult();
  return 0;
}
