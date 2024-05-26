#pragma once
// tensorrt




#include<NvInfer.h>
// cuda
#include<cuda_runtime.h>
#include<stdio.h>
#include <thrust/sort.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<device_atomic_functions.h>
// opencv
#include<opencv2/opencv.hpp>
// cpp std
#include<algorithm>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<sstream>
#include<vector>
#include<chrono>

#include <spdlog/spdlog.h>

class SPDLOG_Logger : public nvinfer1::ILogger {

    void log(Severity severity, const char *msg) noexcept override {
        if (severity == Severity::kINFO) {
            SPDLOG_INFO("[TRT] {}", msg);
        } else if (severity == Severity::kWARNING) {
            SPDLOG_WARN("[TRT] {}", msg);
        } else if (severity == Severity::kERROR) {
            SPDLOG_ERROR("[TRT] {}", msg);

        } else if (severity == Severity::kINTERNAL_ERROR) {
            SPDLOG_CRITICAL("[TRT] {}", msg);
        } else {
            SPDLOG_DEBUG("[TRT] {}", msg);
        }


    }

};
