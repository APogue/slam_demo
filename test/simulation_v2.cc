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
#include "transformation.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
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


Eigen::Quaterniond quat_pos(Eigen::Quaterniond q){
    if (q.w() < 0) {
        q.w() = (-1)*q.w();
        q.x() = (-1)*q.x();
        q.y() = (-1)*q.y();
        q.z() = (-1)*q.z();
    }
    return q;
};

void create_csv(std::vector<State*> state_vec, std::string file_path){
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

class SimSLAM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // parameters
  int state_length =1000;
  double sigma_g_c = 6.0e-4;
  double sigma_a_c = 2.0e-3;
  double del_t = 0.01;
  double T = 0.0; // time
  double r = 10.0; // circle radius x-y plane
  double w = .76; // angular velocity
  double r_z = (1.0/20)*r;
  double w_z = (2.3)*w;
  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);

  // create containers
  std::vector<State*> gt_vec_;
  std::vector<IMU*> imu_vec_;
  std::vector<State*> dr_vec_;

public:
  bool SetGroundTruth() {
    for (unsigned i = 0; i < state_length; i++) {
        Eigen::Matrix<double, 3, 3> Tr;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Eigen::Quaterniond quat;

        pos(0) = r * cos(w * T);
        pos(1) = r * sin(w * T);
        pos(2) = r_z * sin(w_z * T);

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
    T = 0.0;
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

  bool SetDeadReckoning(){
  // Dead reckoning -> est. trajectory
    Eigen::Vector3d p_dr;
    Eigen::Vector3d v_dr;
    Eigen::Quaterniond q_dr;
    T = 0;
    p_dr= gt_vec_.at(0)->GetPositionBlock()->estimate();
    v_dr= gt_vec_.at(0)->GetVelocityBlock()->estimate();
    q_dr= gt_vec_.at(0)->GetRotationBlock()->estimate();
    dr_vec_.push_back(new State(T));
    dr_vec_.at(0)->GetPositionBlock()->setEstimate(p_dr);
    dr_vec_.at(0)->GetVelocityBlock()->setEstimate(v_dr);
    dr_vec_.at(0)->GetRotationBlock()->setEstimate(q_dr);
    for (unsigned i=1; i<state_length; i++) {
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
      T = T + del_t;
    }
    return true;
  }
  bool OutputTrajectoryResult(){

    create_csv(gt_vec_, "trajectory.csv");
    create_csv(dr_vec_, "trajectory_obs.csv");

    return true;
  }
};


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  SimSLAM traj_problem;

  traj_problem.SetGroundTruth();
  traj_problem.GetIMUData();
  traj_problem.SetDeadReckoning();
  traj_problem.OutputTrajectoryResult();

  return 0;
}
