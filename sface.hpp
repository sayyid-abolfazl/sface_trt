#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"  //       توابع مهم تنسور تی آر تی 


using namespace nvinfer1;

// تعریف کلاس مدل مورد نیاز
class sface {
    public:
        sface();
        sface(const std::string engine_file_path);
        void imagePreProcess(cv::Mat& img, cv::Mat& img_resized);   // توابع پیش پردازش که میاد اندازه و کانال رنگی عکس رو تغییر میده آماده میکنه برای مدل
        void imagePostProcess(float* output, cv::Mat& img);  // اگر خروجی مدلمون تصویر داشته باشد این میاد تصویر رو میسازد آماده میکنه
        void blobFromImage(cv::Mat& img, float* blob); //  این میاد تصاویر رو به اعداد اعشاری تبدیل میکند بین -1 +1 سپس ترتیب چنل رنگ ابعادش رو درست میکنه
        void doInference(IExecutionContext& context, float* input, float* output);   // ارسال ورودی ها به سمت مدل که این میاد محیط اجرای برای استنتاج مدل اجرا میکند 
        std::vector<float> infer(cv::Mat& img);   //  این هم برای دریافت ورودی و سپس نتایجی داشته باشیم برای نمایش به صورت عکس در متغییر دوم ذخیره میشود  
        ~sface(); // وظیفه ازاد سازی منابع دارد

    private:
        static const int INPUT_H = 112;
        static const int INPUT_W = 112;
        static const int INPUT_SIZE = 3 * INPUT_H * INPUT_W;
        static const int OUTPUT_SIZE = 128;   // حجم داده خروجی (تعداد ویژگی‌ها)



        const char *INPUT_BLOB_NAME = "data";
        const char *OUTPUT_BLOB_NAME = "fc1";

        int inputIndex;
        int outputIndex;

        IRuntime* runtime = nullptr;
        ICudaEngine* engine = nullptr;
        IExecutionContext* context = nullptr;

        float* input = new float[INPUT_SIZE];
        float* output = new float[OUTPUT_SIZE];
};
