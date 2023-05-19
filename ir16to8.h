
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define INTERRATIO 10 // for find peaks
#define MIN_THRE 100  // for Normalize4Display mode == 2

using namespace cv;
using namespace std;

std::vector<int> mean_filter(std::vector<int> vec, int scale)
{
    int sum = 0;
    std::vector<int> data_filter;

    int halfscale = int(scale / 2);
    for (int i = 0; i < halfscale; i++)
    {
        data_filter.push_back(vec.at(i));
    }
    for (int i = 0; i < vec.size() - scale; ++i)
    {
        for (int j = 0; j < scale; ++j)
        {
            sum += vec.at(i + j);
        }
        data_filter.push_back((int)(sum / scale));
        sum = 0;
    }
    for (int j = vec.size() - (scale - halfscale); j < vec.size(); j++)
    {
        data_filter.push_back(vec.at(j));
    }

    // test
    // if (vec.size() == data_filter.size())
    //{
    //	cout << "scale is right" << endl;
    // }
    //

    return data_filter;
}

static void findpeaks(
    std::vector<int> data,
    std::vector<int> &peakval,
    std::vector<int> &location,
    float ratio,
    int smoothtimes)
{
    std::vector<int> data_filter, peakval_ori, location_ori;
    data_filter = data;

    int smoothindex = 0;
    while (smoothindex < smoothtimes)
    {
        data_filter = mean_filter(data_filter, 5);
        smoothindex++;
    }

    std::vector<int> data_filter_inter;

    for (int i = 0; i < data_filter.size() - 1; i++)
    {
        for (int j = 0; j < INTERRATIO; j++)
        {
            data_filter_inter.push_back((data_filter.at(i) * (INTERRATIO - j) + data_filter.at(i + 1) * j) / INTERRATIO);
        }
    }

    int val_loop = data_filter_inter.at(1);
    int flag_up = 1;
    int maxValue = *max_element(data_filter_inter.begin(), data_filter_inter.end());
    int peakthre = int(maxValue * ratio);
    for (int i = 1; i < data_filter_inter.size(); i++)
    {
        if (data_filter_inter.at(i) < peakthre)
        {
            continue;
        }
        if (data_filter_inter.at(i) > val_loop)
        {
            val_loop = data_filter_inter.at(i);
            flag_up = 1;
        }
        if (data_filter_inter.at(i) < val_loop)
        {
            if (flag_up == 1)
            {
                peakval_ori.push_back(val_loop);
                location_ori.push_back(floor(i / 10));
            }
            val_loop = data_filter_inter.at(i);
            flag_up = 0;
        }
    }
    for (int i = 0; i < location_ori.size(); i++)
    {
        int location_begin, location_end;
        location_begin = location_ori.at(i) - 5;
        location_end = location_ori.at(i) + 5;
        if (location_begin < 0)
        {
            location_begin = 0;
        }
        if (location_end >= data.size())
        {
            location_end = data.size() - 1;
        }

        auto maxPosition = max_element(data.begin() + location_begin, data.begin() + location_end);
        peakval.push_back(int(*maxPosition));
        location.push_back(int(maxPosition - data.begin()));
    }
}

static void MyGammaCorrection(
    cv::Mat &src,
    cv::Mat &dst,
    float fGamma,
    float ratio)
{
    // build look up table
    uchar lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f * ratio);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
    case 1: // �Ҷ�ͼ�����
    {

        cv::MatIterator_<uchar> it, end;
        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
            //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
            *it = lut[(*it)];
    }
    break;
    case 3: // ��ɫͼ�����
    {

        cv::MatIterator_<cv::Vec3b> it, end;
        for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
        {
            //(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
            //(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
            //(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
            (*it)[0] = lut[((*it)[0])];
            (*it)[1] = lut[((*it)[1])];
            (*it)[2] = lut[((*it)[2])];
        }
    }
    break;
    default:
        break;
    }
}

void EnhanceIR(const cv::Mat &input, cv::Mat &output)
{
    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
    kernel.at<float>(1, 1) = 5;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    cv::filter2D(input, output, input.depth(), kernel);
}

// mode = 0或其他 : 模糊后，求最大最小值，再拉伸
// mode = 1 : 根据直方图，去除固定数量极值点，再拉伸
// mode = 2 : 根据直方图，自适应选取拉伸范围，再拉伸
int Normalize4Display(
    const cv::Mat &ref, // 找极值的参考图（防止src有黑边）
    const cv::Mat &src, // 原始图像, CV_16UC1
    cv::Mat &result,    // 拉伸后图像, CV_8UC1
    const int mode,     // 选取不同算法
    const bool gamma,   // true: 开启gamma校正
    const int th1,      // 去除极小值的数量，仅mode为1才有效
    const int th2,      // 去除极大值的数量，仅mode为1才有效
    const bool enhance, // true: 开启laplacian锐化
    const double bri)   // 亮度调整，默认值1.0
{
    // 首先校验输入的合法性
    // minMaxLoc : check histSize == 0 ?
    double minv = 0.;
    double maxv = 0.;
    cv::minMaxLoc(ref, &minv, &maxv);
    int histSize = (int(maxv) - int(minv));
    if (histSize == 0)
    {
        static cv::Mat tmp = cv::Mat::zeros(src.size(), CV_8UC1);
        tmp.copyTo(result);
        // APP_INFO_WARN("histSize is 0, return -1, in function Normalize4Display");
        return -1;
    }

    // mode = 2 : 根据直方图自适应选取拉伸范围
    if (mode == 2)
    {
        float hranges[2] = {(float)minv, (float)maxv};
        const float *ranges[1] = {hranges};
        cv::Mat imshowIR_float;
        src.convertTo(imshowIR_float, CV_32FC1, 1, 0);
        cv::Mat hist;
        cv::calcHist(&imshowIR_float, 1, 0, cv::Mat(), hist, 1, &histSize, ranges); // ����ͼ��ֱ��ͼ

        std::vector<int> histval, peakval, location;

        for (int i = 0; i < histSize; i++)
        {
            histval.push_back(int(hist.at<float>(i)));
        }
        findpeaks(histval, peakval, location, 0.5, 2);

        std::vector<int> half_locs;
        for (int i = 0; i < location.size(); i++)
        {
            std::vector<int> half_locs_tem, half_vals_tem, minushalf_floop;
            for (int j = 0; j < histSize; j++)
            {
                minushalf_floop.push_back(int(peakval.at(i) / 2) - abs(int(histval.at(j)) - int(peakval.at(i) / 2)));
            }
            int maxValue = *max_element(minushalf_floop.begin(), minushalf_floop.end());
            // cout << maxValue << endl;
            findpeaks(minushalf_floop, half_vals_tem, half_locs_tem, 0.7, 1);
            for (int j = 1; j < half_locs_tem.size(); j++)
            {
                if ((half_locs_tem.at(j - 1) < location.at(i)) && (half_locs_tem.at(j) > location.at(i)))
                {
                    half_locs.push_back(half_locs_tem.at(j - 1));
                    half_locs.push_back(half_locs_tem.at(j));
                    break;
                }
            }
            while (half_locs.size() < 2 * (i + 1))
            {
                half_locs.push_back(location.at(i));
            }
        }
        std::vector<int> peakssum;
        int locs_left, locs_right;
        for (int i = 0; i < peakval.size(); i++)
        {
            locs_left = half_locs.at(2 * i);
            locs_right = half_locs.at(2 * i + 1);
            int sum_val = 0;
            for (int j = locs_left; j <= locs_right; j++)
            {
                sum_val = sum_val + histval.at(j);
            }
            peakssum.push_back(sum_val);
        }
        int loc_final, peak_final, half_left_final, half_right_final;
        auto maxsum_pos = max_element(peakssum.begin(), peakssum.end());
        loc_final = maxsum_pos - peakssum.begin();
        peak_final = peakval.at(loc_final);
        half_left_final = half_locs.at(2 * loc_final);
        half_right_final = half_locs.at(2 * loc_final + 1);

        int min_offset = half_left_final, max_offset = half_right_final;

        for (; min_offset >= 0; min_offset--)
        {
            if (histval.at(min_offset) < MIN_THRE)
            {
                break;
            }
        }
        for (; max_offset < histval.size(); max_offset++)
        {
            if (histval.at(max_offset) < MIN_THRE)
            {
                break;
            }
        }
        imshowIR_float = (imshowIR_float - (minv + min_offset)) / ((minv + max_offset) - (minv + min_offset)) * 255;

        if (gamma == true)
        {
            cv::Mat imshowIR;
            imshowIR_float.convertTo(imshowIR, CV_8UC1, 1, 0);
            MyGammaCorrection(imshowIR, result, 1.2, 0.5);
        }
        else
        {
            imshowIR_float.convertTo(result, CV_8UC1, 1, 0);
        }
    }
    else if (mode == 1)
    {
        // calcHist : to acquire new min and max
        float hranges[2] = {(float)minv, (float)maxv};
        const float *ranges[1] = {hranges};
        cv::Mat hist;
        const int dims = 1;
        cv::calcHist(&ref, 1, 0, cv::Mat(), hist, dims, &histSize, ranges);
        int sum_low = 0;
        int minv_low = 0;
        for (int i = 0; i < histSize; i++)
        {
            sum_low += hist.at<float>(i);
            if (sum_low > th1)
            {
                minv_low = i;
                break;
            }
        }
        double minv_new = minv + (double)minv_low;
        int sum_high = 0;
        int maxv_high = 0;
        for (int j = 0; j < histSize; j++)
        {
            sum_high += hist.at<float>(histSize - j - 1);
            if (sum_high > th2)
            {
                maxv_high = j;
                break;
            }
        }
        double maxv_new = maxv - (double)maxv_high;
        // assert : max_new > min_new
        if (maxv_new <= minv_new)
            maxv_new = minv_new + 1.;
        // norm : with new min and max
        cv::Mat src_float;
        src.convertTo(src_float, CV_32FC1, 1, 0);
        src_float = (src_float - minv_new) / (maxv_new - minv_new);
        src_float.convertTo(result, CV_8UC1, 255 * bri, 0);
    }
    else
    {
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(ref, &minv, &maxv);
        cv::Mat src_norm;
        src.convertTo(src_norm, CV_32FC1);
        src_norm = (src_norm - minv) / ((maxv - minv));
        src_norm.convertTo(result, CV_8UC1, 255, 0);
    }
    if (enhance == true)
    {
        cv::Mat result_enhance = result.clone();
        EnhanceIR(result_enhance, result);
    }
    return 0;
}

int Normalize4Display2(
    const cv::Mat &ref, // 找极值的参考图（防止src有黑边）
    const cv::Mat &src, // 原始图像, CV_16UC1
    cv::Mat &result     // 拉伸后图像, CV_8UC1
    )                   // 亮度调整，默认值1.0
{
    // 首先校验输入的合法性
    // minMaxLoc : check histSize == 0 ?
    double minv = 0.;
    double maxv = 0.;
    cv::minMaxLoc(ref, &minv, &maxv);
    int histSize = (int(maxv) - int(minv));
    if (histSize == 0)
    {
        static cv::Mat tmp = cv::Mat::zeros(src.size(), CV_8UC1);
        tmp.copyTo(result);
        return -1;
    }

    cv::minMaxLoc(ref, &minv, &maxv);
    cv::Mat src_norm;
    src.convertTo(src_norm, CV_32FC1);
    src_norm = (src_norm - minv) / ((maxv - minv));
    src_norm.convertTo(result, CV_8UC1, 255, 0);

    return 0;
}

int Normalize4Display3(
    const cv::Mat &src,    // 原始图像, CV_16UC1
    cv::Mat &dst_image_raw // 拉伸后图像, CV_8UC1
    )                      // 亮度调整，默认值1.0
{

    double minv = 9999999.0, maxv = 0.0;
    double *minp = &minv;
    double *maxp = &maxv;

    cv::GaussianBlur(src, dst_image_raw, Size(17, 17), 15, 15);
    cv::minMaxIdx(dst_image_raw, minp, maxp);

    double hist_min; // minv
    double hist_max; // maxv
    hist_max = maxv;
    hist_min = minv;

    double range_d = hist_max - hist_min;
    if (range_d < 10)
        range_d = 10;
    double alpha = 254.0 / range_d;
    double beta = -254.0 * (hist_min) / range_d;

    src.convertTo(dst_image_raw, CV_8UC1, alpha, beta);

    return 0;
}