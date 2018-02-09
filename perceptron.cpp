#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>

#include<boost/algorithm/string/classification.hpp>
#include<boost/algorithm/string/split.hpp>
#include<boost/lexical_cast.hpp>    

#define MAX_FEATURE_DIMENSION 1024
#define MAX_SAMPLE_NUM 1024

int calc_activation(double x[], double weights[], int feature_num) {
    double dot = 0;
    for(int i = 0; i <= feature_num; i++) {
        dot += x[i] * weights[i];
    }
    if (dot > 0) {
        return 1;
    }
    return 0;
}

void perceptron_train(double X[][MAX_FEATURE_DIMENSION], int y[], double weights[], int feature_num, int sample_num, double alpha, int iterate_num) {

    for (int i=0; i<iterate_num; i++) {

        double delta[MAX_FEATURE_DIMENSION] = {0};
        for (int m=0; m<sample_num; m++) {
            //计算感知机输出
            int output = calc_activation(X[m], weights, feature_num);

            //与标注数据比较
            // h == y[m]
            // h != y[m]
            // 不需要比较了 w_new = wi + alpha * (y-output) * xi

            for (int n=0; n <= feature_num; n++) {
                delta[n] += alpha * (y[m] - output) * X[m][n];
            }
        }

        for (int n=0; n <= feature_num; n++) {
            weights[n] += delta[n];
        }

        for (int n=0; n < feature_num; n++) {
            printf("%.3lf ", weights[n]);
        }
        printf("%.3lf\n", weights[feature_num]);
    }
}


double weights[MAX_FEATURE_DIMENSION] = {0};
int y[MAX_SAMPLE_NUM] = {0};
double X[MAX_SAMPLE_NUM][MAX_FEATURE_DIMENSION] = {0};


int main() {
    char fname[] = "train.dat";
    std::ifstream in(fname);
    std::string line;

    getline(in, line);
    std::vector<std::string> dest;
    boost::split(dest, line, boost::is_any_of(" "), boost::token_compress_on);
    int feature_num = boost::lexical_cast<int>(dest[0]);
    int sample_num  = boost::lexical_cast<int>(dest[1]);
    double alpha = boost::lexical_cast<double>(dest[2]);
    int iterate_num = boost::lexical_cast<int>(dest[3]);
    std::cout << feature_num << " " << sample_num << " " << alpha << " " << iterate_num << std::endl;

    getline(in, line);
    dest.clear();
    boost::split(dest, line, boost::is_any_of(" "), boost::token_compress_on);
    for (int i=0; i <= feature_num; i++) {
        weights[i] = boost::lexical_cast<double>(dest[i]);
    }
    for (int i=0; i <= feature_num; i++) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;

    for (int i=0; i<sample_num; i++) {
        getline(in, line);

        std::vector<std::string> dest1;
        boost::split(dest1, line, boost::is_any_of(" "), boost::token_compress_on);

        for (int j=1; j <= feature_num; j++) {
            X[i][j] = boost::lexical_cast<double>(dest1[j-1]);
        }
        //偏移特征x0 = 1
        X[i][0] = 1;
        y[i] = boost::lexical_cast<double>(dest1[feature_num]); 
    }

    for (int i=0; i<sample_num; i++) {

        for (int j=0; j <= feature_num; j++) {
            std::cout << X[i][j] << " ";
        }
        std::cout << y[i] << std::endl;
    }


    in.close();

    perceptron_train(X, y, weights, feature_num, sample_num, alpha, iterate_num);

    return 0;
}
