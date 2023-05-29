#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>


using namespace cv;
using namespace std;


bool isInside(int row, int col, Mat img) {
    if (row >= 0 && row <= img.rows && col >= 0 && col <= img.cols)
        return true;
    else
        return false;
}

Mat_<Vec3i> padImage(Mat_<Vec3i> img) {
    Mat_<Vec3i> paddedImage(img.rows - (img.rows % 8), img.cols - (img.cols % 8));
    for (int i = 0; i < img.rows - (img.rows % 8); i++) {
        for (int j = 0; j < img.cols - (img.cols % 8); j++) {
            paddedImage(i, j) = img(i, j);
        }
    }
    return paddedImage;
}

cv::Mat convertToVec3i(const cv::Mat& vec3bMat) {
    CV_Assert(vec3bMat.type() == CV_8UC3); // make sure the input matrix has 3 channels of unsigned char

    cv::Mat vec3iMat(vec3bMat.size(), CV_32SC3); // create an output matrix with 3 channels of 32-bit integers

    for (int row = 0; row < vec3bMat.rows; ++row) {
        for (int col = 0; col < vec3bMat.cols; ++col) {
            cv::Vec3b bgr = vec3bMat.at<cv::Vec3b>(row, col); // get the BGR color at the current pixel

            // convert the BGR color from unsigned char to int and store it in the output matrix
            vec3iMat.at<cv::Vec3i>(row, col) = cv::Vec3i(static_cast<int>(bgr[0]), static_cast<int>(bgr[1]), static_cast<int>(bgr[2]));
        }
    }

    return vec3iMat;
}


Mat_<Vec3i> downSampling(Mat_<Vec3b> src) {
    Mat_<Vec3b> dst(src.rows, src.cols);
    cvtColor(src, dst, COLOR_BGR2YCrCb);
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            int averageCb = 0;
            int averageCr = 0;
            for (int k = i; k < i + 2; k++) {
                for (int t = j; t < j + 2; t++) {
                    averageCb += dst(i, j)[1];
                    averageCr += dst(i, j)[0];
                }
            }
            averageCr /= 4;
            averageCb /= 4;
            for (int k = i; k < i + 2; k++) {
                for (int t = j; t < j + 2; t++) {
                    dst(i, j)[1] = averageCb;
                    dst(i, j)[0] = averageCr;
                }
            }
        }
    }
    return convertToVec3i(dst);
//    imshow("asd", dst);
//    waitKey();
}

Mat_<Vec3i> makeBlock(Mat_<Vec3i> img, int x, int y) {
    Mat_<Vec3i> res(8, 8);
    for (int i = x; i < x + 8; i++) {
        for (int j = y; j < y + 8; j++) {
            res(i - x, j - y) = img(i, j);
            res(i - x, j - y)[0] -= 128;
            res(i - x, j - y)[1] -= 128;
            res(i - x, j - y)[2] -= 128;
        }
    }
    return res;
}

vector<Mat_<Vec3i>> intoBlocks(Mat_<Vec3i> src) {
    vector<Mat_<Vec3i>> blocks;
    for (int i = 0; i < src.rows; i += 8) {
        for (int j = 0; j < src.cols; j += 8) {
            blocks.push_back(makeBlock(src, i, j));
        }
    }
    return blocks;
}

double c(int u) {
    if (u == 0) {
        return 1.0 / (sqrt(2));
    } else if (u > 0) {
        return 1;
    }
}

//Mat_<Vec3i> dct(Mat_<Vec3b> block) {
//    Mat_<Vec3b> dct(8, 8);
//    vector<float> y[2];
//    vector<float> cB[2];
//    vector<float> cR[2];
//    for (int i = 0 ; i < 8; i++) {
//        for (int j = 0; j < 8; j++) {
//            y[0].push_back(block(i, j)[0]);
//            cB[0].push_back(block(i, j)[1]);
//            cR[0].push_back(block(i, j)[2]);
//        }
//    }
//    cv::dct(y[0], y[1]);
//    cv::dct(cB[0], cB[1]);
//    cv::dct(cR[0], cR[1]);
//    int k = 0;
//    for(int i = 0; i < 8; i++) {
//        for (int j = 0; j < 8; j++) {
//            dct(i, j)[0] = floor (y[1][k]);
//            dct(i, j)[1] = floor (cB[1][k]);
//            dct(i, j)[2] = floor (cR[1][k]);
//            k++;
//        }
//    }
//    return dct;
//}
//
//Mat_<Vec3b> inverseDCT(Mat_<Vec3b> dct) {
//    Mat_<Vec3b> block(8, 8);
//    vector<float> y[2];
//    vector<float> cB[2];
//    vector<float> cR[2];
//    for (int i = 0 ; i < 8; i++) {
//        for (int j = 0; j < 8; j++) {
//            y[0].push_back(dct(i, j)[0]);
//            cB[0].push_back(dct(i, j)[1]);
//            cR[0].push_back(dct(i, j)[2]);
//        }
//    }
//    cv::idct(y[0], y[1]);
//    cv::idct(cB[0], cB[1]);
//    cv::idct(cR[0], cR[1]);
//    int k = 0;
//    for(int i = 0; i < 8; i++) {
//        for (int j = 0; j < 8; j++) {
//            block(i, j)[0] = floor (y[1][k]);
//            block(i, j)[1] = floor (cB[1][k]);
//            block(i, j)[2] = floor (cR[1][k]);
//            k++;
//        }
//    }
//    return block;
//}
//
Mat_<Vec3i> dct(Mat_<Vec3i> block) {
    Mat_<Vec3i> dct(8, 8);
    dct.setTo(0);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double sum[3] = {0, 0, 0};
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    sum[0] += block(x, y)[0] * cos((2 * x + 1) * i * CV_PI / 16) * cos((2 * y + 1) * j * CV_PI / 16);
                    sum[1] += block(x, y)[1] * cos((2 * x + 1) * i * CV_PI / 16) * cos((2 * y + 1) * j * CV_PI / 16);
                    sum[2] += block(x, y)[2] * cos((2 * x + 1) * i * CV_PI / 16) * cos((2 * y + 1) * j * CV_PI / 16);
                }
            }
            dct(i, j)[0] = (int) (1.0 / 4 * c(i) * c(j) * sum[0]);
            dct(i, j)[1] = (int) (1.0 / 4 * c(i) * c(j) * sum[1]);
            dct(i, j)[2] = (int) (1.0 / 4 * c(i) * c(j) * sum[2]);
        }
    }
    return dct;
}

Mat_<Vec3i> inverseDCT(Mat_<Vec3i> dct) {
    Mat_<Vec3i> block(8, 8);
    block.setTo(0);
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double sum[3] = {0, 0, 0};
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    sum[0] += c(i) * c(j) * dct(i, j)[0] * cos((2 * x + 1) * i * CV_PI / 16) *
                              cos((2 * y + 1) * j * CV_PI / 16);
                    sum[1] += c(i) * c(j) * dct(i, j)[1] * cos((2 * x + 1) * i * CV_PI / 16) *
                              cos((2 * y + 1) * j * CV_PI / 16);
                    sum[2] += c(i) * c(j) * dct(i, j)[2] * cos((2 * x + 1) * i * CV_PI / 16) *
                              cos((2 * y + 1) * j * CV_PI / 16);
                }
            }
            block(x, y)[0] = (int) (1.0 / 4 * sum[0]);
            block(x, y)[1] = (int) (1.0 / 4 * sum[1]);
            block(x, y)[2] = (int) (1.0 / 4 * sum[2]);
        }
    }
    return block;
}

vector<vector<int>> qMatrix = {{16, 11, 10, 16, 24,  40,  51,  61},
                               {12, 12, 14, 19, 26,  58,  60,  65},
                               {14, 13, 16, 24, 40,  57,  69,  56},
                               {14, 17, 22, 29, 51,  87,  80,  62},
                               {18, 22, 37, 56, 68,  109, 103, 77},
                               {24, 35, 55, 64, 81,  104, 113, 92},
                               {49, 64, 78, 87, 103, 121, 120, 101},
                               {72, 92, 95, 98, 112, 100, 103, 99}};

Mat_<Vec3i> quantization(Mat_<Vec3i> src) {
    Mat_<Vec3i> dst(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            dst(i, j)[0] = round(src(i, j)[0] / qMatrix[i][j]);
            dst(i, j)[1] = round(src(i, j)[1] / qMatrix[i][j]);
            dst(i, j)[2] = round(src(i, j)[2] / qMatrix[i][j]);
        }
    }
    return dst;
}

vector<Vec3i> zigzagScan(const Mat_<Vec3i> matrix) {
    vector<Vec3i> result;
    int m = matrix.rows;
    int n = matrix.cols;
    int i = 0, j = 0;
    bool isUp = true;

    for (int k = 0; k < m * n; k++) {
        result.emplace_back(matrix(i, j));

        if (isUp) {
            if (i == 0 || j == n - 1) {
                isUp = false;
                if (j == n - 1) i++;
                else j++;
            } else {
                i--;
                j++;
            }
        } else {
            if (j == 0 || i == m - 1) {
                isUp = true;
                if (i == m - 1) j++;
                else i++;
            } else {
                i++;
                j--;
            }
        }
    }

    return result;
}

std::vector<std::pair<int, int>> runLengthEncode(const std::vector<int> &input) {
    std::vector<std::pair<int, int>> output;
    int current_value = input[0];
    int count = 1;

    for (int i = 1; i < input.size(); i++) {
        if (input[i] == current_value && count < 255) {
            count++;
        } else {
            output.emplace_back(current_value, count);
            count = 1;
            current_value = input[i];
        }
    }

    output.emplace_back(current_value, count);

    return output;
}

std::ofstream file("compressed.txt", std::ios::binary);
std::ifstream infile("compressed.txt", std::ios::in | std::ios::binary);

void saveToBinaryFile(vector<pair<int, int>> zigzag, const string &filename) {

    if (!file) {
        printf("Couldn't open the file");
        return;
    }
    for (const auto &p: zigzag) {
        char bytes[2];
        bytes[0] = (char) p.first;
        bytes[1] = (char) p.second;
        file.write(bytes, 2);
    }
}

void saveIntToBinaryFile(int x, const string &filename) {
    if (!file) {
        printf("Couldn't open the file");
        return;
    }

    file.write(reinterpret_cast<const char*>(&x), sizeof(int));

}

vector<Mat_<Vec3i>> pipelineForCompressing() {
    Mat_<Vec3i> firstImage = imread("images/sample1.bmp", IMREAD_COLOR);
    auto img = padImage(firstImage);
    auto mat = downSampling(img);
    auto blocks = intoBlocks(mat);
    //cout << 1;
    saveIntToBinaryFile(img.rows, "compressed.txt");
    saveIntToBinaryFile(img.cols, "compressed.txt");
    saveIntToBinaryFile(blocks.size(), "compressed.txt");
    //cout << 2;
    vector<Mat_<Vec3i>> deq;
    for (int i = 0; i < blocks.size(); i++) {
        //cout << blocks[i] << endl;
        Mat_<Vec3i> blockAfterDct = dct(blocks[i]);
        //cout << blockAfterDct << endl;
        auto dctBlockAfterQuantization = quantization(blockAfterDct);
        auto compressedVector = zigzagScan(dctBlockAfterQuantization);
        deq.push_back(blocks[i]);
        vector<int> yComponent;
        vector<int> cbComponent;
        vector<int> crComponent;
        for (const auto &j: compressedVector) {
            yComponent.push_back(j[0]);
            cbComponent.push_back(j[1]);
            crComponent.push_back(j[2]);
        }
        auto yEncoded = runLengthEncode(yComponent);
        auto cBEncoded = runLengthEncode(cbComponent);
        auto cREncoded = runLengthEncode(crComponent);
        saveIntToBinaryFile(yEncoded.size(), "compressed.txt");
        saveToBinaryFile(yEncoded, "compressed.txt");
        saveIntToBinaryFile(cBEncoded.size(), "compressed.txt");
        saveToBinaryFile(cBEncoded, "compressed.txt");
        saveIntToBinaryFile(cREncoded.size(), "compressed.txt");
        saveToBinaryFile(cREncoded, "compressed.txt");
    }
    file.close();
    return deq;
}

std::vector<int> runLengthDecode(const std::vector<std::pair<int, int>> &input) {
    std::vector<int> output;

    for (const auto &p: input) {
        int value = p.first;
        int count = p.second;

        for (int i = 0; i < count; i++) {
            output.push_back(value);
        }
    }

    return output;
}

Mat_<Vec3i> zigzagReverse(const vector<Vec3i> &zigzag, const int m, const int n) {
    Mat_<Vec3i> matrix(m, n);
    int i = 0, j = 0;
    bool isUp = true;
    for (int k = 0; k < zigzag.size(); k++) {
        matrix(i, j) = zigzag[k];

        if (isUp) {
            if (i == 0 || j == n - 1) {
                isUp = false;
                if (j == n - 1) i++;
                else j++;
            } else {
                i--;
                j++;
            }
        } else {
            if (j == 0 || i == m - 1) {
                isUp = true;
                if (i == m - 1) j++;
                else i++;
            } else {
                i++;
                j--;
            }
        }
    }
    return matrix;
}

Mat_<Vec3i> dequantization(Mat_<Vec3i> src) {
    Mat_<Vec3i> dst(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            dst(i, j)[0] = src(i, j)[0] * qMatrix[i][j];
            dst(i, j)[1] = src(i, j)[1] * qMatrix[i][j];
            dst(i, j)[2] = src(i, j)[2] * qMatrix[i][j];
        }
    }
    return dst;
}

Mat_<Vec3i> combineBlocks(vector<Mat_<Vec3i>> blocks, int rows, int cols) {
    Mat_<Vec3i> img(rows, cols);
    int blockIndex = 0;
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j += 8) {
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    auto pixel = blocks[blockIndex](x, y);
                    pixel[0] += 128;
                    pixel[1] += 128;
                    pixel[2] += 128;
                    img(x + i, y + j) = pixel;
                }
            }
            blockIndex++;
        }
    }
    return img;
}

cv::Mat convertToVec3b(const cv::Mat& vec3iMat) {
    CV_Assert(vec3iMat.type() == CV_32SC3); // make sure the input matrix has 3 channels of 32-bit integers

    cv::Mat vec3bMat(vec3iMat.size(), CV_8UC3); // create an output matrix with 3 channels of unsigned char

    for (int row = 0; row < vec3iMat.rows; ++row) {
        for (int col = 0; col < vec3iMat.cols; ++col) {
            cv::Vec3i bgr = vec3iMat.at<cv::Vec3i>(row, col); // get the BGR color at the current pixel

            // convert the BGR color from int to unsigned char and store it in the output matrix
            vec3bMat.at<cv::Vec3b>(row, col) = cv::Vec3b(static_cast<uchar>(bgr[0]), static_cast<uchar>(bgr[1]), static_cast<uchar>(bgr[2]));
        }
    }

    return vec3bMat;
}

float getCompressionRatio(String uncompressedPath) {
    cout << uncompressedPath;
    ifstream uncompressedFile(uncompressedPath, ios::binary | ios::ate);
    streampos uncompressedFileSize = uncompressedFile.tellg();
    uncompressedFile.close();

    streampos compressedFileSize = infile.tellg();
    cout << uncompressedFileSize << endl << (float) compressedFileSize << endl;
    infile.close();
    return (float) uncompressedFileSize / (float)compressedFileSize;
}

vector<Mat_<Vec3i>> pipelineForDecompressing() {
    vector<Mat_<Vec3i>> blocks;
    int rows, cols, blockSize = -1;
    bool alreadyRead = false;
    int k = 0;
    vector<Mat_<Vec3i>> deq;
    while (!infile.eof()) {
        if (k == blockSize) break;
        if (!alreadyRead) {
            infile.read(reinterpret_cast<char*>(&rows), sizeof(int));
            infile.read(reinterpret_cast<char*>(&cols), sizeof(int));
            infile.read(reinterpret_cast<char*>(&blockSize), sizeof(int));
            alreadyRead = true;
        }
        int ySize, cbSize, crSize;
        infile.read(reinterpret_cast<char*>(&ySize), sizeof(int));
        int x, count;
        vector<pair<int, int>> yEncoded;
        for (int i = 0; i < ySize * 2; i += 2) {
            char bytes[2];
            infile.read(bytes, 2);
            int x = (int) bytes[0];
            int count = (int) bytes[1];
            yEncoded.push_back({x, count});
        }
        infile.read(reinterpret_cast<char*>(&cbSize), sizeof(int));
        vector<pair<int, int>> cbEncoded;
        for (int i = 0; i < cbSize * 2; i += 2) {
            char bytes[2];
            infile.read(bytes, 2);
            int x = (int) bytes[0];
            int count = (int) bytes[1];
            cbEncoded.push_back({x, count});
        }
        infile.read(reinterpret_cast<char*>(&crSize), sizeof(int));
        vector<pair<int, int>> crEncoded;
        for (int i = 0; i < crSize * 2; i += 2) {
            char bytes[2];
            infile.read(bytes, 2);
            int x = (int) bytes[0];
            int count = (int) bytes[1];
            crEncoded.push_back({x, count});
        }
        auto decodedY = runLengthDecode(yEncoded);
        auto decodedCb = runLengthDecode(cbEncoded);
        auto decodedCr = runLengthDecode(crEncoded);
        vector<Vec3i> zigzag;
        for (int i = 0; i < 64; i++) {
            zigzag.push_back({decodedY[i], decodedCb[i], decodedCr[i]});
        }
        auto matAfterZigZag = zigzagReverse(zigzag, 8, 8);
        auto matAfterDequantization = dequantization(matAfterZigZag);
        auto matAfterInverseDct = inverseDCT(matAfterDequantization);
        //cout << matAfterInverseDct<<endl;
        deq.push_back(matAfterInverseDct);
        blocks.push_back(matAfterInverseDct);
        k++;
    }
    const Mat_<Vec3i> &show = combineBlocks(blocks, rows, cols);
    //Mat_<Vec3i> dst;
    const Mat &mat = convertToVec3b(show);
    cvtColor(mat, mat, COLOR_YCrCb2BGR);
    //resize(mat, dst, Size(400, 312), 0, 0, INTER_CUBIC);
    imshow("asd", mat);
    waitKey();
    cout << getCompressionRatio("images/sample1.bmp");
    return deq;
}


int main() {
    auto compressing = pipelineForCompressing();
    auto decompressing = pipelineForDecompressing();

    return 0;
}
