#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <iostream>
#include <fstream>
#include <istream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

using namespace std;

typedef struct MOT
{
    cv::Mat state_of_1obj;
    int id;
    int tracked_i;
    int missing;
}MOT;

typedef struct Match
{
    int id;
    bool toofar;
}Match;

class hungarian
{
    void stp1(int &step);//reduce with the minima of row and column
    void stp2(int &step);
    void stp3(int &step);
    void stp4(int &step);
    void stp5(int &step);
    void stp6(int &step);
    void stp7();
    void find_a_zero(int& row, int& col);
    bool star_in_row(int row);
    void find_star_in_row(int row, int& col);
    void find_min(double& minval);
    void find_star_in_col(int col, int& row);
    void find_prime_in_row(int row, int& col);
    void augment_path();
    void clear_covers();
    void erase_primes();

    int step = 1;
    Eigen::MatrixXd cost, mask, path, copy;
    vector<int> cover_row;
    vector<int> cover_col;
    int path_row_0, path_col_0, path_count;

    void cost_generate(vector<cv::Point> detected, vector<cv::Point> previous);

public:
    hungarian();
    ~hungarian();
    vector<Match> solution(vector<cv::Mat>measured, vector<MOT>previous);//return the corresponding ids
    vector<Match> id_match;
};

#endif // HUNGARIAN_H
