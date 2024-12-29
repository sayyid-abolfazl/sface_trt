#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>


#include "sface.hpp"


#define DEVICE 0  // GPU id






#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

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

// 

// 

int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);

    if (argc != 5 || std::string(argv[2]) != "-i") {                 
        std::cerr << " arguments not right ! " << "\n";
        std::cerr << " For Example : " << "\n";
        std::cerr << "./demo ../models/model.engine -i ../images/test.png ../images/2.png " << "\n";
        return -1;
    }

    const std::string engine_file_path = argv[1];
    const std::string q = argv[3];
    const std::string t = argv[4];




    sface sample = sface(engine_file_path);


    cv::Mat target = cv::imread("111.jpg");
    cv::Mat query = cv::imread("444.jpg");
    std::vector<float> vec_s1 = sample.infer(target);
    std::vector<float> vec_s2 = sample.infer(query);

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
                
    
    }; // 128 ویژگی

    std::vector<float> vec_onnx1 = {
        
        -1.7179822 ,  0.6564609 , -1.0647364 , -0.70782745, -0.14286849,
        -0.10902017, -1.3021411 , -0.33553702,  1.0994564 , -0.66910225,
        0.05673549,  1.9792243 ,  1.0743947 , -0.21698561,  0.6745935 ,
        0.54311633,  0.7595187 , -0.38423416,  0.03660099, -0.49724004,
        0.5056483 ,  2.1010637 , -1.4586555 ,  0.8838126 ,  0.1918744 ,
        -1.8427391 ,  0.05449065, -0.2661895 ,  0.48029292,  1.5273198 ,
        0.32496956,  1.8302406 , -0.78103656,  0.7674867 , -1.4364387 ,
        0.43543446, -0.3235675 , -0.6117327 ,  0.9160278 , -2.3339293 ,
        0.7782307 ,  0.03402354, -1.3195915 ,  0.56168467,  0.27369   ,
        -0.813949  , -0.8264483 ,  0.529851  ,  0.3350946 ,  0.61294115,
        1.2581551 , -0.31351095,  1.8375161 , -1.7013233 , -1.1085179 ,
        -0.80826986,  0.42728865,  0.79817885, -0.9445191 ,  0.24375781,
        -0.43600875, -1.0872229 , -0.37747476, -1.5050787 , -0.58066195,
        -0.79607844,  0.9655771 ,  0.7575069 ,  1.1719381 ,  0.5031031 ,
        -1.2840176 ,  0.19543956, -0.08010055,  1.3876262 ,  2.7067566 ,
        -3.734546  ,  0.58853537, -0.8721188 ,  0.40296343,  0.06157439,
        -1.169652  ,  0.21362472,  0.23140194,  1.1391896 ,  0.71354496,
        0.52315104,  0.20296276,  0.25904092, -1.5249878 , -1.0655183 ,
        0.03047626,  0.87265754,  1.4249079 ,  0.05582605, -0.73617584,
        -0.80756515, -0.50670624,  0.45515552,  1.4988574 ,  0.8100662 ,
        -0.12381744, -0.65324736, -2.0939891 , -0.22820728,  0.25445655,
        -1.0035957 ,  1.2556089 , -0.03834247,  0.94604707,  0.5135448 ,
        0.15220915, -1.7795776 ,  0.8268679 ,  0.66912365,  0.25936642,
        -0.21946414,  0.04142863,  0.6880018 ,  0.87044007,  0.20214759,
        -0.50374526,  0.7281609 , -0.63916385,  0.88356113, -0.39560845,
        0.8856474 , -2.7403674 , -0.7886186
                    
    }; // 128 ویژگی

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