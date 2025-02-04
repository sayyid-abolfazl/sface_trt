#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include "sface.hpp"


#define DEVICE 0  // GPU id








// روش اول: استفاده از فرمول استاندارد کسینوسی
float cosineSimilarityStandard(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dotProduct = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
        return 0.0f; // جلوگیری از تقسیم بر صفر
    }

    return dotProduct / (magnitude1 * magnitude2);
}

// روش دوم: استفاده از کتابخانه STL برای جمع‌زدن مقادیر
float cosineSimilaritySTL(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dotProduct = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0f);
    float magnitude1 = std::sqrt(std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0f));
    float magnitude2 = std::sqrt(std::inner_product(vec2.begin(), vec2.end(), vec2.begin(), 0.0f));

    if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
        return 0.0f; // جلوگیری از تقسیم بر صفر
    }

    return dotProduct / (magnitude1 * magnitude2);
}

// روش سوم: محاسبه کسینوسی با دقت بیشتر (حذف مقادیر بسیار کوچک)
float cosineSimilaritySafe(const std::vector<float>& vec1, const std::vector<float>& vec2, float epsilon = 1e-6) {
    float dotProduct = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    if (magnitude1 < epsilon || magnitude2 < epsilon) {
        return 0.0f; // جلوگیری از تقسیم بر صفر یا مقادیر نزدیک به صفر
    }

    return dotProduct / (magnitude1 * magnitude2);
}

// روش چهارم: استفاده از توابع ریاضی پیشرفته برای بردارهای بزرگ
float cosineSimilarityAdvanced(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dotProduct = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;

    #pragma omp parallel for reduction(+:dotProduct, magnitude1, magnitude2)
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (magnitude1 * magnitude2);
}

// روش پنجم: تفاضل عادی نرمال شده
float normalizedDifference(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float sumAbsDiff = 0.0f;
    float sumMagnitudes = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        sumAbsDiff += std::abs(vec1[i] - vec2[i]);
        sumMagnitudes += std::abs(vec1[i]) + std::abs(vec2[i]);
    }

    if (sumMagnitudes == 0.0f) {
        return 0.0f;
    }

    return 1.0f - (2.0f * sumAbsDiff / sumMagnitudes);
}

// روش ششم: تفاضل اقلیدسی نرمال شده
float normalizedEuclidean(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float sumSquaredDiff = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        sumSquaredDiff += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }

    return 1.0f - std::sqrt(sumSquaredDiff) / std::sqrt(vec1.size());
}

// روش هفتم: ضریب همبستگی نرمال شده
float normalizedCorrelation(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float mean1 = std::accumulate(vec1.begin(), vec1.end(), 0.0f) / vec1.size();
    float mean2 = std::accumulate(vec2.begin(), vec2.end(), 0.0f) / vec2.size();

    float numerator = 0.0f;
    float denominator1 = 0.0f;
    float denominator2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        numerator += (vec1[i] - mean1) * (vec2[i] - mean2);
        denominator1 += (vec1[i] - mean1) * (vec1[i] - mean1);
        denominator2 += (vec2[i] - mean2) * (vec2[i] - mean2);
    }

    if (denominator1 == 0.0f || denominator2 == 0.0f) {
        return 0.0f;
    }

    return numerator / std::sqrt(denominator1 * denominator2);
}



class YuNet
{
  public:
    YuNet(const std::string& model_path,
          const cv::Size& input_size,
          const float conf_threshold,
          const float nms_threshold,
          const int top_k,
          const int backend_id,
          const int target_id)
    {
        _detector = cv::FaceDetectorYN::create(
            model_path, "", input_size, conf_threshold, nms_threshold, top_k, backend_id, target_id);
    }

    void setInputSize(const cv::Size& input_size)
    {
        _detector->setInputSize(input_size);
    }

    void setTopK(const int top_k)
    {
        _detector->setTopK(top_k);
    }

    cv::Mat infer(const cv::Mat& image)
    {
        cv::Mat result;
        _detector->detect(image, result);
        return result;
    }

  private:
    cv::Ptr<cv::FaceDetectorYN> _detector;
};







int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);

    if (argc != 5 || std::string(argv[2]) != "-i") {                 
        std::cerr << " arguments not right ! " << "\n";
        std::cerr << " For Example : " << "\n";
        std::cerr << "./demo ../models/model.engine -i ../images/test.png ../images/2.png " << "\n";
        return -1;
    }


    
    // https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    auto face_detector = YuNet(
    "face_detection_yunet_2023mar.onnx", cv::Size(320, 320), 0.7 , 0.3 , 1000, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);



    
    const std::vector<std::pair<int, int>> backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX,  cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN,   cv::dnn::DNN_TARGET_NPU}
        };








    const std::string engine_file_path = argv[1];
    const std::string q = argv[3];
    const std::string t = argv[4];




    sface sface_obj = sface(engine_file_path);


    cv::Mat target = cv::imread(t);
    cv::Mat query = cv::imread(q);
    cv::imshow("target",target);
    cv::imshow("query",query);
    cv::waitKey(0);





    static std::vector<cv::Scalar> landmark_colors{
        cv::Scalar(255, 0, 0),   // right eye
        cv::Scalar(0, 0, 255),   // left eye
        cv::Scalar(0, 255, 0),   // nose tip
        cv::Scalar(255, 0, 255), // right mouth corner
        cv::Scalar(0, 255, 255)  // left mouth corner
    };



    
    // Detect single face in target image
    face_detector.setInputSize(query.size());
    face_detector.setTopK(1000);
    cv::Mat faces_query = face_detector.infer(query);




    cv::Mat  face_crop_query;

    if (faces_query.total() > 0 && faces_query.cols >= 15) {
        float x = faces_query.at<float>(0, 0);
        float y = faces_query.at<float>(0, 1);
        float w = faces_query.at<float>(0, 2);
        float h = faces_query.at<float>(0, 3);
        float confidence = faces_query.at<float>(0, 14);

        cv::Rect2f original_face_box(x, y, w, h);

        // تنظیم مستطیل به اندازه تصویر
        original_face_box &= cv::Rect2f(0, 0, query.cols, query.rows);

        if (original_face_box.width > 0 && original_face_box.height > 0) {
            face_crop_query = query(original_face_box);
            

            // رسم مستطیل روی تصویر اصلی
            cv::rectangle(query, original_face_box, cv::Scalar(0, 255, 0), 2);

            // محاسبه لندمارک‌ها
            std::vector<cv::Point2f> original_landmarks(5);
            for (int j = 0; j < 5; ++j) {
                float original_x = faces_query.at<float>(0, 4 + 2 * j);
                float original_y = faces_query.at<float>(0, 5 + 2 * j);

                if (original_x >= 0 && original_x < query.cols &&
                    original_y >= 0 && original_y < query.rows) {
                    original_landmarks[j] = cv::Point2f(original_x, original_y);
                    cv::circle(query, original_landmarks[j], 2, landmark_colors[j], -1);
                }
            }

            cv::imshow("query", face_crop_query);
            cv::waitKey(0);



        } else {
            std::cerr << "Invalid face box!" << std::endl;
        }
    } else {
        std::cerr << "No face detected!" << std::endl;
    }



    face_detector.setInputSize(target.size());
    face_detector.setTopK(1000);
    cv::Mat faces_target = face_detector.infer(target);


    cv::Mat face_crop_target;

    if (faces_target.total() > 0 && faces_target.cols >= 15) {
        float x = faces_target.at<float>(0, 0);
        float y = faces_target.at<float>(0, 1);
        float w = faces_target.at<float>(0, 2);
        float h = faces_target.at<float>(0, 3);
        float confidence = faces_target.at<float>(0, 14);

        cv::Rect2f original_face_box(x, y, w, h);

        // تنظیم مستطیل به اندازه تصویر
        original_face_box &= cv::Rect2f(0, 0, target.cols, target.rows);

        if (original_face_box.width > 0 && original_face_box.height > 0) {
            face_crop_target = target(original_face_box);

            // رسم مستطیل روی تصویر اصلی
            cv::rectangle(target, original_face_box, cv::Scalar(0, 255, 0), 2);

            // محاسبه لندمارک‌ها
            std::vector<cv::Point2f> original_landmarks(5);
            for (int j = 0; j < 5; ++j) {
                float original_x = faces_target.at<float>(0, 4 + 2 * j);
                float original_y = faces_target.at<float>(0, 5 + 2 * j);

                if (original_x >= 0 && original_x < target.cols &&
                    original_y >= 0 && original_y < target.rows) {
                    original_landmarks[j] = cv::Point2f(original_x, original_y);
                    cv::circle(target, original_landmarks[j], 2, landmark_colors[j], -1);
                }
            }

            cv::imshow("target", face_crop_target);
            cv::waitKey(0);



        } else {
            std::cerr << "Invalid face box!" << std::endl;
        }
    } else {
        std::cerr << "No face detected!" << std::endl;
    }





    std::vector<float> vec_s1 = sface_obj.infer(face_crop_query);
    std::vector<float> vec_s2 = sface_obj.infer(face_crop_target);



    //You can post the output of the onnx model here to check the difference in the outputs.


    // بردارهای ویژگی نمونه
    std::vector<float> vec_onnx2 = {
        
        -0.6653271 , -2.7673187 , -0.23382767,  1.149541  , -0.7340615 ,
        -0.128987  ,  0.17279188,  2.4389706 , -0.46219474,  1.1263875 ,
        -1.9969009 , -0.34240744, -1.1973597 ,  1.4731086 , -1.695758  ,
        -1.0363989 , -1.1396216 , -2.914533  ,  0.29993814,  0.5750167 ,
        0.9120602 ,  0.54088044,  0.7939873 , -0.19628045,  0.26954088,
        -0.37881723,  0.53219146,  0.52044654,  1.5490209 , -0.16775514,
        0.28989914,  1.8574605 ,  0.62409043, -0.50787663,  0.4257083 ,
        -0.6575221 , -1.6562384 , -1.8734355 , -0.45478776, -0.5372416 ,
        1.017202  ,  0.5531686 , -0.88766617,  0.27554712, -0.17140014,
        0.37407604, -0.5793743 , -0.7182468 , -0.869754  , -2.4433749 ,
        -2.0673978 ,  0.33117187,  0.9268872 , -0.30351108, -0.5441622 ,
        1.2150955 ,  1.794834  ,  0.17300957, -0.8900786 , -0.9620178 ,
        -1.3901591 ,  0.36766016, -0.48622575, -0.991331  ,  0.66936356,
        -1.5811976 , -1.2146747 ,  0.9689572 , -0.10913387, -0.09859667,
        -0.15183519,  1.2548866 ,  0.4363814 , -1.1811914 ,  1.635252  ,
        0.1758948 , -0.9795277 , -0.7293735 ,  0.93355525, -0.73128515,
        1.2729601 ,  0.8110027 , -1.3586754 ,  0.12754472, -0.20499809,
        2.1042836 , -0.3124517 , -0.31703684,  0.5102345 ,  0.940918  ,
        -0.38189107,  3.8702846 ,  0.15849344,  0.2778694 ,  1.1442939 ,
        -0.66167426, -0.17311613,  1.528527  ,  0.42863983,  0.5438388 ,
        -0.27148056, -0.22276506, -0.5151833 , -0.2819142 ,  0.5029464 ,
        -1.2934926 , -2.5658855 ,  0.03840356, -0.43388578,  1.5854716 ,
        0.84706163,  0.6069427 , -1.5668564 ,  0.5753083 ,  2.3445764 ,
        -1.3055    , -2.01115   ,  0.26059034,  1.0996912 , -1.8880942 ,
        -0.6834755 , -0.53702193,  1.458859  , -0.88288474, -0.5808352 ,
        0.61602205,  0.64711094,  0.05647586
                        
    
    }; 

    std::vector<float> vec_onnx1 = {
        
        3.2790530e-01, -1.5365446e+00, -1.3242102e+00,  1.2409792e+00,
        -2.7258158e-01, -2.0406487e+00,  4.9704936e-01,  1.9078279e+00,
        -1.0051534e+00, -7.9201776e-01, -3.0298296e-01,  1.6219306e+00,
        1.6060572e+00,  9.5780432e-01,  1.1979878e+00, -1.7008569e+00,
        -8.2805490e-01, -1.3175385e+00,  2.1399561e-01,  6.9383204e-01,
        -6.5262511e-02, -5.3778476e-01,  1.0744743e+00, -2.2741408e+00,
        -2.7306964e+00,  4.9025226e-01, -7.3746783e-01,  3.6385757e-01,
        5.3154302e-01, -7.7065915e-01,  4.7597572e-01, -4.8986650e-01,
        -9.0130115e-01, -2.0708339e+00,  4.4152823e-01, -2.3998962e-01,
        6.8964928e-02,  4.5314556e-01, -5.5004275e-01,  3.8459998e-01,
        1.0561351e+00,  4.6152782e-01,  1.1736318e+00,  1.6482894e+00,
        1.0772864e+00, -1.5275799e-01, -1.2691997e+00, -1.3586960e+00,
        1.3368955e+00,  9.0550238e-01, -2.0675287e-01,  2.2661140e+00,
        5.9332275e-01,  1.1782950e+00,  1.2461269e+00,  3.2238719e-01,
        6.7247242e-01, -8.3125257e-01,  6.8761572e-02, -2.7081497e+00,
        -7.6467299e-01,  7.2135109e-01, -1.5446901e+00, -4.0035555e-01,
        -8.1679732e-02, -1.9907393e+00,  8.1078970e-01,  6.1983550e-01,
        1.3161994e+00,  1.7966273e+00, -1.6318296e+00,  8.7770122e-01,
        -9.4180292e-01,  3.7487742e-01, -5.1890320e-01, -2.2657459e+00,
        -5.3627688e-01, -1.1255981e+00, -1.1895188e+00, -4.5786881e-01,
        2.3753619e+00, -1.7044790e+00, -4.9245644e-01, -8.2061106e-01,
        -2.4532299e+00,  7.5733435e-01,  9.5817554e-01, -9.3460781e-03,
        -2.3034678e+00, -3.6875024e-01,  1.1810551e+00,  3.3984312e-01,
        8.1700087e-01,  2.6740214e-01,  5.5383903e-01, -4.5838475e-01,
        -6.5805435e-01,  1.7217162e+00, -6.1314893e-01,  5.1607120e-01,
        9.5685345e-01, -2.3502645e+00, -1.3855140e+00, -9.5718163e-01,
        -1.0941375e+00, -9.9444028e-04, -1.0826866e-01,  6.3040143e-01,
        1.3179220e-01, -2.1676111e+00,  1.0939851e+00,  2.4895388e-01,
        -1.4269400e+00,  5.1890349e-01, -8.8770592e-01, -7.5672638e-01,
        4.5518523e-01, -1.1810297e+00,  5.4373610e-01, -2.1511212e-01,
        -3.3364172e+00,  6.9221562e-01,  2.4890668e+00, -1.2017430e+00,
        -2.3514113e-01,  1.2106668e+00, -2.4143462e+00,  7.0786905e-01
                    
    }; 

    std::cout << "\n\n--------------------------- SFace ONNX demo.py : -------------------- " << "\n";


    std::cout << "Cosine Similarity (Standard): " << cosineSimilarityStandard(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Cosine Similarity (STL): " << cosineSimilaritySTL(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Cosine Similarity (Safe): " << cosineSimilaritySafe(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Cosine Similarity (Advanced): " << cosineSimilarityAdvanced(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Normalized Difference: " << normalizedDifference(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Normalized Euclidean: " << normalizedEuclidean(vec_onnx2, vec_onnx1) << "\n";
    std::cout << "Normalized Correlation: " << normalizedCorrelation(vec_onnx2, vec_onnx1) << "\n";

    std::cout << "--------------------------- SFace : -------------------- " << "\n";


    std::cout << "Cosine Similarity (Standard): " << cosineSimilarityStandard(vec_s1, vec_s2) << "\n";
    std::cout << "Cosine Similarity (STL): " << cosineSimilaritySTL(vec_s1, vec_s2) << "\n";
    std::cout << "Cosine Similarity (Safe): " << cosineSimilaritySafe(vec_s1, vec_s2) << "\n";
    std::cout << "Cosine Similarity (Advanced): " << cosineSimilarityAdvanced(vec_s1, vec_s2) << "\n";
    std::cout << "Normalized Difference: " << normalizedDifference(vec_s1, vec_s2) << "\n";
    std::cout << "Normalized Euclidean: " << normalizedEuclidean(vec_s1, vec_s2) << "\n";
    std::cout << "Normalized Correlation: " << normalizedCorrelation(vec_s1, vec_s2) << "\n";


    return 0;
}
