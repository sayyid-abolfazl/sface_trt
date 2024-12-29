#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"


#include "sface.hpp"


using namespace nvinfer1;

// تابع چک که اگر اروری چیزی خورد معلوم بشه برای ساده شدن برنامه 
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kERROR)
            std::cout << msg << std::endl;
    }
} gLogger;


// سازنده پیش فرض

sface::sface() {}



// سازنده پارامتر دار که میاد فایل مدل تی آر تی میخواند
sface::sface(const std::string engine_file_path) {
    char *trtModelStream = nullptr;   // پوینتر برای ذخیره محتویات فایل مدل 
    size_t size = 0;  // اندازه فایل مدل بر حسب بایت


    // باز کردن فایل مدل به صورت باینری اگر نتوانست باز کند خطا میدهد
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);    //  اشاره گر فایل  را به انتهای فایل می برد
        size = file.tellg();    //  موقعیت اشاره گر اندازه فایل را ذخیره میکند
        file.seekg(0, file.beg); // اشاره گر را دوباره به ابتدای فایل برمیگرداند
        trtModelStream = new char[size];  // یک آرایه پویا به اندازه فایل مدل ایجاد و محتویات فایل به این آرایه کپی میشود
        assert(trtModelStream);  // 
        file.read(trtModelStream, size); 
        file.close();  // اتمام کار 
    } else {
        std::cerr << "could not open engine!" << std::endl;
        return;
    }


    // رابط اصلی تی آر تی رو ایجاد میکند اگر اجرا نشد به خطا میخورد 
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // trtModelStream محتوای فایل ذخیره شده در 
    // deserializeCudaEngine از طریق 
    // یک موتور کودا تبدیل میشود 
    // اگر عملیات انجام نشد به خطا میخورد
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 

    // ایجاد محیط اجرا برای انجام عملیات اینفرنس یا استنتاج مدل
    context = engine->createExecutionContext();
    assert(context != nullptr);

    // آزاد سازی حافظه
    delete[] trtModelStream;

    // بررسی ورودی و خروجی های مدل که دوتا زده شده 
    assert(engine->getNbBindings() == 2);

    // INPUT_BLOB_NAME ورودی مدل با استفاده از نام ها شناسایی میشوند
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

    // OUTPUT_BLOB_NAME خروجی مدل با استفاده از نام ها شناسایی میشوند
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
}

// تمام منابع اختصاص داده شده در کلاس را آزاد میکند
sface::~sface() {
    delete context;
    delete engine;
    delete runtime;

    delete input;
    delete output;
}
// تبدیل آر جی بی به بی جی آر
// void sface::imagePreProcess(cv::Mat& img, cv::Mat& img_resized) {
//     cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
//     cv::Size dsize = cv::Size(INPUT_W, INPUT_H);
//     cv::resize(img_resized, img_resized, dsize);
// }


void sface::imagePreProcess(cv::Mat& img, cv::Mat& img_resized) {
    // رنگ تصویر را تغییر ندهید
    cv::Size dsize = cv::Size(INPUT_W, INPUT_H);
    cv::resize(img, img_resized, dsize); // فقط اندازه تصویر را تغییر می‌دهیم
}

// خروجی را تبدیل به عکس میکند که نیازی بهش نداریم
void sface::imagePostProcess(float* output, cv::Mat& img) {
    img.create(cv::Size(INPUT_H, INPUT_W), CV_8UC3);
    for (int i = 0; i < OUTPUT_SIZE; i ++) {
        int w = i % INPUT_H;
        int h = (i / INPUT_W) % INPUT_H;
        int c = i / INPUT_H / INPUT_W;

        float pixel = output[i] * 0.5 + 0.5;
        if (pixel < 0) pixel = 0;
        if (pixel > 1) pixel = 1;
        pixel *= 255;

        img.at<cv::Vec3b>(h, w)[c] = pixel;
    }
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
}

// تبدیل تصویر به آرایه استاندارد


/*
پیکسل‌های تصویر به اعداد اعشاری تبدیل می‌شوند
این اعداد نرمال‌سازی می‌شوند تا در بازه [-1, 1] قرار گیرند
داده‌ها به ترتیب مناسب برای ورودی مدل ذخیره می‌شوند
(ترتیب CHW )


*/
void sface::blobFromImage(cv::Mat& img, float* input) {
    int channels = 3;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < INPUT_H; h++) {
            for (int w = 0; w < INPUT_W; w++) {
                input[c * INPUT_W * INPUT_H + h * INPUT_W + w] =
                    ((float)img.at<cv::Vec3b>(h, w)[c] / 255.0 - 0.5) / 0.5;
            }
        }
    }
}



// وظیفه ی اجرای عملیات پیش بینی اینفرنس روی مدل دارد یعنی داده ها  رو از سی پی یو به جی پی یو انتقال میدهد 
void sface::doInference(IExecutionContext& context, float* input, float* output) {

    // ایجاد بافر که همان حافظه ای روی جی پی یو 
    void* buffers[2];
    // بافر محل ذخیره داده های ورودی 
    CHECK(cudaMalloc(&buffers[inputIndex], INPUT_SIZE * sizeof(float)));  //cudaMalloc ==>>   یعنی حافظه را از جی پی یو

    // بافر محل ذخیره خروجی مدل
    CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float))); 



    /*
    یک استریم کودا ایجاد میشود کهیک کانالی تشکیل میدهد جی پی یو میتواند به طور غیر همزمان آسینکرون  عملیات انجام دهد
    یعنی چندتا کار به طور همزمان انجام شود بدون اینکه منتظر عملیات قبلی بماند
    */
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // داده های پیش پردازش شده از حافظه هاست که همون سی پی یو به حافظه دیوایس  جی پی یو کپی میشود
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    
    // اجرای مدل روی جی پی یو
    context.enqueueV2(buffers, stream, nullptr);
    
    // خروجی مدل را از حافظه هاست به جی پی یو انتقال میدهد که به طور غیر همزمان انجام میشود و الان سی پی یو به نتایج دسترسی دارد
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    // منتظر اتمام کار فرآیند های بالا میماند تا عملیات ها انجام شوند
    cudaStreamSynchronize(stream);


    // استریم کودا حذف میشود تا منابع آزاد شوند
    cudaStreamDestroy(stream);
    // حافظه هایی که برای خروجی و ورودی جی پی یو ازاد میشود
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}




// عکس رو بهش میدیدم و خروجی را برای ما نمایش میدهد
std::vector<float> sface::infer(cv::Mat& img) {
    cv::Mat img_resized;
    imagePreProcess(img, img_resized);
    blobFromImage(img_resized, input);
    doInference(*context, input, output);

    // چاپ 128 ویژگی خروجی
    // std::cout << "Feature vector : " << output << "\n";


    // تبدیل خروجی به یک بردار برای بازگشت
    std::vector<float> featureVector(output, output + OUTPUT_SIZE);


    std::cout << "Feature vector (128-dimensional output):" << "\n";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << output[i];
        if (i != OUTPUT_SIZE - 1) {
            std::cout << ", "; // جدا کردن ویژگی‌ها با کاما
        }
    }

    return featureVector; // بازگشت بردار ویژگی

    std::cout << "\n"; // پایان خط
}