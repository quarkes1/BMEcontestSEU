#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <iomanip>

// 配置参数：可根据测试需求修改
const int SC_NUM = 2;        // 正常组(SC)被试数量
const int ST_NUM = 3;        // 障碍组(ST)被试数量
const int PART_NUM = 5;      // 每被试的Part数，固定5个（贴合竞赛）
const int DURATION_PER_PART = 300;  // 每Part的时长(秒)，300s=5分钟（可改）
const int SAMPLING_RATE = 100;      // 采样频率100Hz
const int FRAME_DURATION = 30;      // 标签帧时长30s

// 生成随机浮点数（模拟脑电电压，范围-1.0 ~ 1.0V）
double getRandomVoltage() {
        return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// 生成随机标签：W(清醒)、R(REM)、1(N1)、2(N2)、3(N3)、?(未知)
char getRandomLabel() {
        int randVal = rand() % 6;
        switch (randVal) {
        case 0: return 'W';
        case 1: return 'R';
        case 2: return '1';
        case 3: return '2';
        case 4: return '3';
        case 5: return '?';
        default: return '?';
        }
}

// 生成单个被试的所有Part文件（脑电+标签）
// type: S=SC(正常), T=ST(障碍); id: 被试编号（如001,002）
void generateSubjectFiles(char type, int id) {
        // 被试ID格式化：SC001、ST701（贴合竞赛的701/702格式，可自定义）
        std::string subjectId = (type == 'S' ? "SC" : "ST") + std::to_string(700 + id);
        int dataPointNum = DURATION_PER_PART * SAMPLING_RATE;  // 每Part脑电数据点数量
        int frameNum = DURATION_PER_PART / FRAME_DURATION;     // 每Part标签帧数

        // 遍历生成5个Part的文件
        for (int part = 1; part <= PART_NUM; part++) {
                // ========== 1. 生成脑电数据文件 ==========
                std::string eegFileName = subjectId + "_EEGFpz Cz_Part_" + std::to_string(part) + "_of_5.txt";
                std::ofstream eegFile(eegFileName);
                if (!eegFile.is_open()) {
                        std::cerr << "创建脑电文件失败：" << eegFileName << std::endl;
                        continue;
                }
                // 写入随机电压值，保留6位小数（模拟实际数据精度）
                eegFile << std::fixed << std::setprecision(6);
                for (int i = 0; i < dataPointNum; i++) {
                        eegFile << getRandomVoltage() << std::endl;
                }
                eegFile.close();
                std::cout << "生成脑电文件：" << eegFileName << std::endl;

                // ========== 2. 生成标签文件 ==========
                std::string labelFileName = subjectId + "_Hypnogram_Data_Part_" + std::to_string(part) + "_of_5.txt";
                std::ofstream labelFile(labelFileName);
                if (!labelFile.is_open()) {
                        std::cerr << "创建标签文件失败：" << labelFileName << std::endl;
                        continue;
                }
                // 按帧生成标签：开始时间 结束时间 持续时间 标签
                int startTime = 0;
                for (int frame = 0; frame < frameNum; frame++) {
                        int endTime = startTime + FRAME_DURATION;
                        char label = getRandomLabel();
                        labelFile << startTime << " " << endTime << " " << FRAME_DURATION << " " << label << std::endl;
                        startTime = endTime;
                }
                labelFile.close();
                std::cout << "生成标签文件：" << labelFileName << std::endl;
        }
        std::cout << "================ 被试" << subjectId << "文件生成完成 ================" << std::endl;
}

int main() {
        // 初始化随机数种子（保证每次运行生成不同数据）
        srand((unsigned int)time(NULL));
        std::cout << "===== 开始生成睡眠脑电模拟数据（竞赛格式）=====" << std::endl;

        // 生成SC正常组被试
        for (int i = 1; i <= SC_NUM; i++) {
                generateSubjectFiles('S', i);
        }
        // 生成ST障碍组被试
        for (int i = 1; i <= ST_NUM; i++) {
                generateSubjectFiles('T', i);
        }

        std::cout << "===== 所有模拟文件生成完成！=====" << std::endl;
        system("pause");  // 防止控制台闪退（Windows）
        return 0;
}