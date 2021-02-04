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

struct IMU {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d gyro_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d accel_ = Eigen::Vector3d(0, 0, 0);

};

struct State {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double t_;
    Eigen::Quaterniond q_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d v_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d p_ = Eigen::Vector3d(0, 0, 0);

};

struct Observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Quaterniond q_obs_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d v_obs_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d p_obs_ = Eigen::Vector3d(0, 0, 0);
};

Eigen::Quaterniond quat_pos(Eigen::Quaterniond q){
    if (q.w() < 0) {
        q.w() = (-1)*q.w();
        q.x() = (-1)*q.x();
        q.y() = (-1)*q.y();
        q.z() = (-1)*q.z();
    }
    return q;
}

int main(int argc, char **argv) {

    // parameters

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
    std::vector<State*> state_vec;
    std::vector<IMU*> imu_vec;
    std::vector<Observation*> obs_vec;

    for (unsigned i=0; i<1000; i++) {
        Eigen::Matrix<double,3,3> Tr;
        State* new_state_ptr = new State;
        new_state_ptr->t_ = T;
        new_state_ptr->p_(0) = r*cos(w*T);
        new_state_ptr->p_(1) = r*sin(w*T);
        new_state_ptr->p_(2) = r_z*sin(w_z*T);

        new_state_ptr->v_(0) = -r*w*sin(w*T);
        new_state_ptr->v_(1) = r*w*cos(w*T);
        new_state_ptr->v_(2) = r_z*w_z*cos(w_z*T);

        Tr << cos(w*T),-sin(w*T), 0,
             sin(w*T), cos(w*T), 0,
                0, 0, 1;

        new_state_ptr-> q_ = Tr;

        state_vec.push_back(new_state_ptr);

        state_vec.at(i)->q_= quat_pos(state_vec.at(i)->q_);
        T = T + del_t;
    };

    // IMU -> est. trajectory
    T = 0.0;
    for (unsigned i=0; i<999; i++) {
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
        new_imu_ptr->accel_= state_vec.at(i)->q_.toRotationMatrix().transpose() * (a_N - gravity) + acc_noise; // expressed in the body frame

        imu_vec.push_back(new_imu_ptr);
        T = T + del_t;
    };

    // Dead reckoning -> est. trajectory
    Observation* new_obs_ptr = new Observation;
    new_obs_ptr->p_obs_= state_vec.at(0)->p_;
    new_obs_ptr->v_obs_= state_vec.at(0)->v_;
    new_obs_ptr->q_obs_= state_vec.at(0)->q_;
    obs_vec.push_back(new_obs_ptr);
    T = 0;
    for (unsigned i=1; i<1000; i++) {
        Observation* new_obs_ptr = new Observation;

        new_obs_ptr->p_obs_ = obs_vec.at(i-1)->p_obs_ + del_t * obs_vec.at(i-1)->v_obs_ + 0.5 * (del_t * del_t)*
                (obs_vec.at(i-1)->q_obs_.toRotationMatrix()*imu_vec.at(i-1)->accel_ + gravity);

        new_obs_ptr->v_obs_ = obs_vec.at(i-1)->v_obs_ + del_t *
                (obs_vec.at(i-1)->q_obs_.toRotationMatrix()*imu_vec.at(i-1)->accel_ + gravity);

        new_obs_ptr->q_obs_ = obs_vec.at(i-1)->q_obs_.normalized()*Exp(del_t * imu_vec.at(i-1)->gyro_);

        obs_vec.push_back(new_obs_ptr);
        obs_vec.at(i)->q_obs_= quat_pos(obs_vec.at(i)->q_obs_);
        T = T + del_t;
    };


    for (unsigned i=0; i<1000; i++) {
        std::cout << i << std::endl;
        std::cout << "q error : " << state_vec.at(i)->q_.w() - obs_vec.at(i)->q_obs_.w()  << std::endl;
        std::cout << "q error : " << state_vec.at(i)->q_.vec() - obs_vec.at(i)->q_obs_.vec()  << std::endl;
        std::cout << "v error : " << state_vec.at(i)->v_ - obs_vec.at(i)->v_obs_ << std::endl;
        std::cout << "p error : " << state_vec.at(i)->p_ - obs_vec.at(i)->p_obs_   << std::endl;

    };


    std::ofstream output_file_traj("trajectory.csv");
    output_file_traj << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (auto & i : state_vec) {

        output_file_traj << std::to_string(i->t_) << ",";
        output_file_traj << std::to_string(i->p_(0)) << ",";
        output_file_traj << std::to_string(i->p_(1)) << ",";
        output_file_traj << std::to_string(i->p_(2)) << ",";
        output_file_traj << std::to_string(i->v_(0)) << ",";
        output_file_traj << std::to_string(i->v_(1)) << ",";
        output_file_traj << std::to_string(i->v_(2)) << ",";
        output_file_traj << std::to_string(i->q_.w()) << ",";
        output_file_traj << std::to_string(i->q_.x()) << ",";
        output_file_traj << std::to_string(i->q_.y()) << ",";
        output_file_traj << std::to_string(i->q_.z()) << std::endl;
    }

    output_file_traj.close();
    std::ofstream output_file_obs("trajectory_obs.csv");
    output_file_obs << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<obs_vec.size(); ++i) {

        output_file_obs << std::to_string(state_vec.at(i)->t_) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->p_obs_(0)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->p_obs_(1)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->p_obs_(2)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->v_obs_(0)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->v_obs_(1)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->v_obs_(2)) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->q_obs_.w()) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->q_obs_.x()) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->q_obs_.y()) << ",";
        output_file_obs << std::to_string(obs_vec.at(i)->q_obs_.z()) << std::endl;
    }

    output_file_obs.close();

    return 0;
}
