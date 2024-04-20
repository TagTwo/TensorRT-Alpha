#include"yolov8_seg.h"
#include"decode_yolov8_seg.h"

using namespace TagTwo::AI;

YOLOv8Seg::YOLOv8Seg(const utils::InitParameter &param) : yolo::YOLO(param) {
    m_output_objects_device = nullptr;
    m_output_objects_width = 39;
    m_output_src_width = 116;
    m_output_obj_area = 1 + param.topK * m_output_objects_width;
    m_output_seg_w = 160 * 160;
    m_output_seg_h = 32;
    int output_objects_size = param.batch_size * m_output_obj_area;
    CHECK(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];

    m_mask160 = cv::Mat::zeros(1, 160 * 160, CV_32F);
    m_mask_eigen160 = Eigen::MatrixXf(1, 160 * 160);
    m_thresh_roi160 = cv::Rect(0, 0, 160, 160);
    m_thresh_roisrc = cv::Rect(0, 0, m_param.src_w, m_param.src_h);
    m_downsample_scale = 160.f / 640;
    m_mask_src = cv::Mat::zeros(m_param.src_h, m_param.src_w, CV_32F);
    m_img_canvas = cv::Mat::zeros(cv::Size(m_param.src_w, m_param.src_h), CV_8UC3);
}

YOLOv8Seg::~YOLOv8Seg() {
    CHECK(cudaFree(m_output_objects_device));
    CHECK(cudaFree(m_output_src_device));
    CHECK(cudaFree(m_output_src_transpose_device));
    CHECK(cudaFree(m_output_seg_device));
    delete[] m_output_objects_host;
    delete[] m_output_seg_host;
    m_output_src_device = nullptr;
}

bool YOLOv8Seg::init(const std::vector<unsigned char> &trtFile) {

    // Set the device index
    if (auto ret = cudaSetDevice(m_param.device_index); ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_param.device_index) +
                      ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    if (trtFile.empty()) {
        return false;
    }
    this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (m_runtime == nullptr) {
        return false;
    }
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            m_runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr) {
        return false;
    }
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr) {
        return false;
    }
    if (m_param.dynamic_batch) {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }

    m_output_dims = this->m_context->getBindingDimensions(2);
    m_total_objects = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1; // 116 * 8400
    for (int i = 1; i < m_output_dims.nbDims; i++) {
        if (m_output_dims.d[i] != 0) {
            m_output_area *= m_output_dims.d[i];
        }
    }

    m_output_seg_dims = this->m_context->getBindingDimensions(1);
    assert(m_param.batch_size <= m_output_seg_dims.d[0]);
    m_output_seg_area = 1; // 32 * 160 * 160
    for (int i = 1; i < m_output_seg_dims.nbDims; i++) {
        if (m_output_seg_dims.d[i] != 0) {
            m_output_seg_area *= m_output_seg_dims.d[i];
        }
    }

    CHECK(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    CHECK(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
    CHECK(cudaMalloc(&m_output_seg_device, m_param.batch_size * m_output_seg_area * sizeof(float)));
    m_output_seg_host = new float[m_param.batch_size * m_output_seg_area];

    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5,
            0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);
    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];
    return true;
}

void YOLOv8Seg::preprocess(const std::vector<cv::Mat> &imgsBatch) {
    resizeDevice(
            m_param.batch_size,
            m_input_src_device,
            m_param.src_w,
            m_param.src_h,
            m_input_resize_device,
            m_param.dst_w,
            m_param.dst_h,
            114,
            m_dst2src
    );
    bgr2rgbDevice(
            m_param.batch_size,
            m_input_resize_device,
            m_param.dst_w,
            m_param.dst_h,
            m_input_rgb_device,
            m_param.dst_w,
            m_param.dst_h
    );
    normDevice(
            m_param.batch_size,
            m_input_rgb_device,
            m_param.dst_w,
            m_param.dst_h,
            m_input_norm_device,
            m_param.dst_w,
            m_param.dst_h,
            m_param
    );
    hwc2chwDevice(
            m_param.batch_size,
            m_input_norm_device,
            m_param.dst_w,
            m_param.dst_h,
            m_input_hwc_device,
            m_param.dst_w,
            m_param.dst_h
    );
}

bool YOLOv8Seg::infer() {
    float *bindings[] = {
            m_input_hwc_device,
            m_output_seg_device,
            m_output_src_device
    };

    bool context = m_context->executeV2((void **) bindings);
    return context;
}


void YOLOv8Seg::postprocess(const std::vector<cv::Mat> &imgsBatch) {
    yolov8seg::transposeDevice(
            m_param,
            m_output_src_device,
            m_total_objects,
            m_output_src_width,
            m_total_objects * m_output_src_width,
            m_output_src_transpose_device,
            m_output_src_width,
            m_total_objects
    );
    yolov8seg::decodeDevice(
            m_param,
            m_output_src_transpose_device,
            m_output_src_width,
            m_total_objects,
            m_output_area,
            m_output_objects_device,
            m_output_objects_width,
            m_param.topK
    );

    nmsDeviceV1(
            m_param,
            m_output_objects_device,
            m_output_objects_width,
            m_param.topK,
            m_output_obj_area
    );
    CHECK(cudaMemcpy(
            m_output_objects_host,
            m_output_objects_device,
            m_param.batch_size * sizeof(float) * m_output_obj_area,
            cudaMemcpyDeviceToHost
    ));
    CHECK(cudaMemcpy(
            m_output_seg_host,
            m_output_seg_device,
            m_param.batch_size * sizeof(float) * m_output_seg_area,
            cudaMemcpyDeviceToHost
    ));
}

void YOLOv8Seg::reset() {
    CHECK(cudaMemset(
            m_output_objects_device,
            0,
            sizeof(float) * m_param.batch_size * m_output_obj_area
    ));
}


void YOLOv8Seg::getObjects(
        std::vector<Object> &objects,
        const size_t imgIndex
) {

    // Reset Canvas
    m_img_canvas.setTo(cv::Scalar(0, 0, 0));


    Eigen::Map<Eigen::MatrixXf> img_seg_(
            m_output_seg_host + imgIndex * m_output_seg_area,
            m_output_seg_w,
            m_output_seg_h
    );

    int m_output_obj_area_bi = imgIndex * m_output_obj_area;


    int num_boxes = std::min(
            static_cast<int>((m_output_objects_host + imgIndex * m_output_obj_area)[0]),
            m_param.topK
    );

    objects.resize(num_boxes);
    objects.clear();
    size_t filledBoxes = 0;
    for (size_t i = 0; i < num_boxes; i++) {

        float *ptr = m_output_objects_host + m_output_obj_area_bi + m_output_objects_width * i + 1;


        auto left = ptr[0];
        auto top = ptr[1];
        auto right = ptr[2];
        auto bottom = ptr[3];
        auto confidence = ptr[4];
        auto label = ptr[5];
        auto one = ptr[6];
        auto obj_seg_data = ptr + 7;

        if (one == 0) {
            continue;
        }

        Eigen::Map<Eigen::MatrixXf> img_ojb_seg_(
                obj_seg_data,
                m_output_seg_h,
                1
        );

        m_mask_eigen160 = img_seg_ * img_ojb_seg_;

        assert(!(m_mask_eigen160.Flags & Eigen::RowMajorBit));

        cv::eigen2cv(m_mask_eigen160, m_mask160);
        cv::exp(-m_mask160, m_mask160);

        m_mask160 = 1.f / (1.f + m_mask160);
        m_mask160 = m_mask160.reshape(1, 160);

        int x_lt_160 = static_cast<int>(std::round(left * m_downsample_scale));
        int y_lt_160 = static_cast<int>(std::round(top * m_downsample_scale));
        int x_rb_160 = static_cast<int>(std::round(right * m_downsample_scale));
        int y_rb_160 = static_cast<int>(std::round(bottom * m_downsample_scale));

        auto roi160 = cv::Rect(
                x_lt_160,
                y_lt_160,
                x_rb_160 - x_lt_160,
                y_rb_160 - y_lt_160
        ) & m_thresh_roi160;

        if (roi160.width == 0 || roi160.height == 0) {
            continue;
        }

        auto x_lt_src = static_cast<int>(std::round(m_dst2src.v0 * left + m_dst2src.v1 * top + m_dst2src.v2));
        auto y_lt_src = static_cast<int>(std::round(m_dst2src.v3 * left + m_dst2src.v4 * top + m_dst2src.v5));
        auto x_rb_src = static_cast<int>(
                std::round(m_dst2src.v0 * right + m_dst2src.v1 * bottom + m_dst2src.v2)
        );
        auto y_rb_src = static_cast<int>(
                std::round(m_dst2src.v3 * right + m_dst2src.v4 * bottom + m_dst2src.v5)
        );

        auto roi_src = cv::Rect(
                x_lt_src,
                y_lt_src,
                x_rb_src - x_lt_src,
                y_rb_src - y_lt_src
        ) & m_thresh_roisrc;

        if (roi_src.width == 0 || roi_src.height == 0) {
            continue;
        }

        cv::Mat mask_instance;
        cv::resize(
                cv::Mat(m_mask160, roi160),
                mask_instance,
                cv::Size(roi_src.width, roi_src.height),
                cv::INTER_LINEAR
        );
        mask_instance = mask_instance > 0.5f;

        objects.emplace_back();
        auto &object = objects.back();
        object.rect = roi_src;
        object.label = static_cast<int>(label);
        object.probability = confidence;
        object.boxMask = std::move(mask_instance);
        filledBoxes++;
    }
    objects.resize(filledBoxes);
}

void YOLOv8Seg::showAndSave(
        const std::vector<std::string> &classNames,
        const int &cvDelayTime,
        const std::vector<cv::Mat> &imgsBatch
) {

    cv::Point bbox_points[1][4];
    const cv::Point *bbox_point0[1] = {bbox_points[0]};
    int num_points[] = {4};

    std::vector<Object> objects;

    for (size_t bi = 0; bi < imgsBatch.size(); bi++) {

        getObjects(objects, bi);

        for (auto &object: objects) {

            const auto &color = utils::Colors::color80.at(object.label);


#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
            // for opencv >=4.7(faster)
            cv::cvtColor(object.boxMask, object.boxMask, cv::COLOR_GRAY2BGR);
            object.boxMask.setTo(color, object.boxMask);
            cv::addWeighted(
                    object.boxMask,
                    0.45,
                    m_img_canvas(object.rect),
                    1.0,
                    0.,
                    m_img_canvas(object.rect)
            );
# else
            // for opencv >=3.2.0
            cv::Mat mask_instance_bgr;
            cv::cvtColor(object.boxMask, mask_instance_bgr, cv::COLOR_GRAY2BGR);
            mask_instance_bgr.setTo(color, object.boxMask);
            cv::addWeighted(
                    mask_instance_bgr,
                    0.45,
                    m_img_canvas(object.rect),
                    1.0,
                    0.,
                    m_img_canvas(object.rect)
            );
#endif


            if (m_param.is_save || m_param.is_show) {

                // label's info
                cv::rectangle(
                        imgsBatch[bi],
                        object.rect,
                        color,
                        2,
                        cv::LINE_AA
                );

                cv::String det_info = m_param.class_names[object.label] + " " + cv::format("%.4f", object.probability);


                bbox_points[0][0] = cv::Point(
                        object.rect.x,
                        object.rect.y
                );

                bbox_points[0][1] = cv::Point(
                        object.rect.x + det_info.size() * m_param.char_width,
                        object.rect.y
                );

                bbox_points[0][2] = cv::Point(
                        object.rect.x + det_info.size() * m_param.char_width,
                        object.rect.y - m_param.det_info_render_width
                );

                bbox_points[0][3] = cv::Point(
                        object.rect.x,
                        object.rect.y - m_param.det_info_render_width
                );


                cv::fillPoly(
                        imgsBatch[bi],
                        bbox_point0,
                        num_points,
                        1,
                        color
                );
                cv::putText(
                        imgsBatch[bi],
                        det_info,
                        bbox_points[0][0],
                        cv::FONT_HERSHEY_DUPLEX,
                        m_param.font_scale,
                        cv::Scalar(255, 255, 255),
                        1,
                        cv::LINE_AA
                );

            }


        }
        if (m_param.is_show) {
            cv::imshow(m_param.winname, imgsBatch[bi] + m_img_canvas);
            cv::waitKey(cvDelayTime);
        }
        if (m_param.is_save) {
            cv::imwrite(m_param.save_path + utils::getTimeStamp() + ".jpg", imgsBatch[bi] + m_img_canvas);
        }
    }
}