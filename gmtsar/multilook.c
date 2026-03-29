#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <time.h>

// 复数类型定义
typedef struct {
    int16_t real;  // 实部
    int16_t imag;  // 虚部
} ComplexInt16;

// 处理参数结构
typedef struct {
    char input_filename[256];
    char output_filename[256];
    int data_width;          // 数据宽度（像素）
    int range_look;          // 距离向降低比例 (2,4,8)
    int azimuth_look;        // 方位向降低比例 (2,4,8,16)
    int data_height;         // 数据高度，从文件大小计算得出
    int cropped_width;       // 裁剪后的宽度（可被range_look整除）
    int cropped_height;      // 裁剪后的高度（可被azimuth_look整除）
    int output_width;        // 输出宽度
    int output_height;       // 输出高度
} ProcessingParams;

// 错误处理
#define ERROR(msg) do { \
    fprintf(stderr, "错误: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
} while(0)

// 获取文件大小
long get_file_size(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    
    return size;
}

// 解析命令行参数
int parse_arguments(int argc, char *argv[], ProcessingParams *params) {
    if (argc != 6) {
        fprintf(stderr, "用法: %s <输入文件> <数据宽度> <距离向降低比例> <方位向降低比例> <输出文件>\n", argv[0]);
        fprintf(stderr, "示例: %s slc.dat 8000 4 8 slc_multi_looked.dat\n", argv[0]);
        fprintf(stderr, "距离向降低比例: 2, 4, 8\n");
        fprintf(stderr, "方位向降低比例: 2, 4, 8, 16\n");
        return 0;
    }
    
    // 设置输入文件名
    strncpy(params->input_filename, argv[1], sizeof(params->input_filename) - 1);
    
    // 设置数据宽度
    params->data_width = atoi(argv[2]);
    if (params->data_width <= 0) {
        fprintf(stderr, "错误: 数据宽度必须为正整数\n");
        return 0;
    }
    
    // 设置距离向降低比例
    params->range_look = atoi(argv[3]);
    if (params->range_look != 2 && params->range_look != 4 && params->range_look != 8) {
        fprintf(stderr, "错误: 距离向降低比例必须为 2, 4, 8\n");
        return 0;
    }
    
    // 设置方位向降低比例
    params->azimuth_look = atoi(argv[4]);
    if (params->azimuth_look != 2 && params->azimuth_look != 4 && 
        params->azimuth_look != 8 && params->azimuth_look != 16) {
        fprintf(stderr, "错误: 方位向降低比例必须为 2, 4, 8, 16\n");
        return 0;
    }
    
    // 设置输出文件名
    strncpy(params->output_filename, argv[5], sizeof(params->output_filename) - 1);
    
    return 1;
}

// 计算输出尺寸并裁剪多余部分
void calculate_dimensions(ProcessingParams *params) {
    // 获取文件大小
    long file_size = get_file_size(params->input_filename);
    if (file_size <= 0) {
        ERROR("无法获取输入文件大小");
    }
    
    // 每个像素占4字节（2字节实部 + 2字节虚部）
    long total_pixels = file_size / 4;
    
    // 计算原始数据高度
    int original_height = total_pixels / params->data_width;
    if (original_height <= 0) {
        ERROR("数据高度计算错误，请检查数据宽度参数");
    }
    
    // 计算裁剪后的尺寸（确保能被降低比例整除）
    params->cropped_width = (params->data_width / params->range_look) * params->range_look;
    params->cropped_height = (original_height / params->azimuth_look) * params->azimuth_look;
    
    // 计算输出尺寸
    params->output_width = params->cropped_width / params->range_look;
    params->output_height = params->cropped_height / params->azimuth_look;
    
    // 验证计算结果
    if (params->output_width <= 0 || params->output_height <= 0) {
        ERROR("降低比例过大，输出尺寸为0");
    }
    
    params->data_height = original_height;
}

// 读取一行复数数据
int read_complex_line(FILE *file, ComplexInt16 *buffer, int width) {
    size_t elements_read = fread(buffer, sizeof(ComplexInt16), width, file);
    return elements_read == width;
}

// 多视处理一行数据（返回处理后的像素数）
int process_range_multi_look(const ComplexInt16 *input_line, 
                           ComplexInt16 *output_line,
                           int input_width, int range_look) {
    int output_width = input_width / range_look;
    
    for (int out_col = 0; out_col < output_width; out_col++) {
        int32_t sum_real = 0;
        int32_t sum_imag = 0;
        
        // 对range_look个像素进行平均
        int start_col = out_col * range_look;
        
        for (int i = 0; i < range_look; i++) {
            ComplexInt16 pixel = input_line[start_col + i];
            sum_real += pixel.real;
            sum_imag += pixel.imag;
        }
        
        // 计算平均值并转换为int16
        output_line[out_col].real = (int16_t)(sum_real / range_look);
        output_line[out_col].imag = (int16_t)(sum_imag / range_look);
    }
    
    return output_width;
}

// 执行多视处理
void perform_multi_looking(ProcessingParams *params) {
    FILE *input_file = fopen(params->input_filename, "rb");
    if (!input_file) {
        ERROR("无法打开输入文件");
    }
    
    FILE *output_file = fopen(params->output_filename, "wb");
    if (!output_file) {
        fclose(input_file);
        ERROR("无法创建输出文件");
    }
    
    printf("开始多视处理...\n");
    printf("原始尺寸: %d x %d\n", params->data_width, params->data_height);
    printf("裁剪尺寸: %d x %d (丢弃多余部分)\n", params->cropped_width, params->cropped_height);
    printf("输出尺寸: %d x %d\n", params->output_width, params->output_height);
    printf("降低比例: 距离向 %dx, 方位向 %dx\n", 
           params->range_look, params->azimuth_look);
    printf("等效视数: %d\n", params->range_look * params->azimuth_look);
    
    // 分配缓冲区
    ComplexInt16 *input_buffer = (ComplexInt16*)malloc(
        params->cropped_width * sizeof(ComplexInt16));
    ComplexInt16 *range_looked_buffer = (ComplexInt16*)malloc(
        params->output_width * sizeof(ComplexInt16));
    
    if (!input_buffer || !range_looked_buffer) {
        fclose(input_file);
        fclose(output_file);
        ERROR("内存分配失败");
    }
    
    // 方位向处理缓冲区
    ComplexInt16 **azimuth_buffer = (ComplexInt16**)malloc(
        params->azimuth_look * sizeof(ComplexInt16*));
    for (int i = 0; i < params->azimuth_look; i++) {
        azimuth_buffer[i] = (ComplexInt16*)malloc(
            params->output_width * sizeof(ComplexInt16));
        if (!azimuth_buffer[i]) {
            ERROR("方位向缓冲区分配失败");
        }
    }
    
    // 处理进度显示
    clock_t start_time = clock();
    int processed_lines = 0;
    int cropped_lines = 0;
    
    // 主处理循环
    for (int out_row = 0; out_row < params->output_height; out_row++) {
        // 读取azimuth_look行并进行距离向多视
        for (int az_idx = 0; az_idx < params->azimuth_look; az_idx++) {
            // 计算输入行号
            int input_row = out_row * params->azimuth_look + az_idx;
            
            // 检查是否超出裁剪高度
            if (input_row >= params->cropped_height) {
                // 用0填充
                memset(azimuth_buffer[az_idx], 0, 
                       params->output_width * sizeof(ComplexInt16));
                continue;
            }
            
            // 定位到正确行和列（考虑宽度裁剪）
            long offset = input_row * params->data_width * sizeof(ComplexInt16);
            fseek(input_file, offset, SEEK_SET);
            
            // 读取裁剪后的宽度数据（丢弃右侧多余部分）
            if (!read_complex_line(input_file, input_buffer, params->cropped_width)) {
                // 如果读取失败，用0填充
                memset(azimuth_buffer[az_idx], 0, 
                       params->output_width * sizeof(ComplexInt16));
                cropped_lines++;
                continue;
            }
            
            // 距离向多视处理
            process_range_multi_look(input_buffer, 
                                    azimuth_buffer[az_idx],
                                    params->cropped_width, 
                                    params->range_look);
        }
        
        // 方位向多视处理
        ComplexInt16 *output_line = (ComplexInt16*)malloc(
            params->output_width * sizeof(ComplexInt16));
        
        for (int col = 0; col < params->output_width; col++) {
            int32_t sum_real = 0;
            int32_t sum_imag = 0;
            
            // 对azimuth_look行进行平均
            for (int az_idx = 0; az_idx < params->azimuth_look; az_idx++) {
                sum_real += azimuth_buffer[az_idx][col].real;
                sum_imag += azimuth_buffer[az_idx][col].imag;
            }
            
            // 计算平均值并转换为int16
            output_line[col].real = (int16_t)(sum_real / params->azimuth_look);
            output_line[col].imag = (int16_t)(sum_imag / params->azimuth_look);
        }
        
        // 写入输出文件
        fwrite(output_line, sizeof(ComplexInt16), params->output_width, output_file);
        
        free(output_line);
        processed_lines++;
        
        // 显示进度
        if (out_row % 100 == 0 || out_row == params->output_height - 1) {
            float progress = (float)(out_row + 1) / params->output_height * 100;
            printf("处理进度: %.1f%% (行 %d/%d)\r", 
                   progress, out_row + 1, params->output_height);
            fflush(stdout);
        }
    }
    
    // 计算处理时间
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\n处理完成！\n");
    printf("处理时间: %.2f 秒\n", elapsed_time);
    printf("处理速率: %.2f 行/秒\n", params->output_height / elapsed_time);
    
    if (cropped_lines > 0) {
        printf("裁剪行数: %d\n", cropped_lines);
    }
    
    // 清理资源
    for (int i = 0; i < params->azimuth_look; i++) {
        free(azimuth_buffer[i]);
    }
    free(azimuth_buffer);
    free(input_buffer);
    free(range_looked_buffer);
    
    fclose(input_file);
    fclose(output_file);
}

// 生成元数据文件
void generate_metadata_file(ProcessingParams *params) {
    char metadata_filename[256];
    strcpy(metadata_filename, params->output_filename);
    strcat(metadata_filename, ".meta");
    
    FILE *meta_file = fopen(metadata_filename, "w");
    if (!meta_file) {
        fprintf(stderr, "警告: 无法创建元数据文件\n");
        return;
    }
    
    fprintf(meta_file, "# SLC多视处理元数据\n");
    fprintf(meta_file, "# 生成时间: %s", ctime(&(time_t){time(NULL)}));
    fprintf(meta_file, "\n");
    
    fprintf(meta_file, "[输入参数]\n");
    fprintf(meta_file, "输入文件: %s\n", params->input_filename);
    fprintf(meta_file, "原始宽度: %d\n", params->data_width);
    fprintf(meta_file, "原始高度: %d\n", params->data_height);
    fprintf(meta_file, "距离向降低比例: %d\n", params->range_look);
    fprintf(meta_file, "方位向降低比例: %d\n", params->azimuth_look);
    fprintf(meta_file, "\n");
    
    fprintf(meta_file, "[裁剪信息]\n");
    fprintf(meta_file, "裁剪后宽度: %d (丢弃右侧 %d 像素)\n", 
            params->cropped_width, params->data_width - params->cropped_width);
    fprintf(meta_file, "裁剪后高度: %d (丢弃底部 %d 行)\n", 
            params->cropped_height, params->data_height - params->cropped_height);
    fprintf(meta_file, "\n");
    
    fprintf(meta_file, "[输出参数]\n");
    fprintf(meta_file, "输出文件: %s\n", params->output_filename);
    fprintf(meta_file, "输出宽度: %d\n", params->output_width);
    fprintf(meta_file, "输出高度: %d\n", params->output_height);
    fprintf(meta_file, "输出数据类型: complex int16 (4字节/像素)\n");
    fprintf(meta_file, "输出文件大小: %.2f MB\n", 
            (params->output_width * params->output_height * 4.0) / (1024 * 1024));
    fprintf(meta_file, "\n");
    
    fprintf(meta_file, "[处理信息]\n");
    fprintf(meta_file, "总降低倍数: %d\n", params->range_look * params->azimuth_look);
    fprintf(meta_file, "等效视数: %d\n", params->range_look * params->azimuth_look);
    fprintf(meta_file, "数据保留率: %.1f%%\n", 
            (float)(params->cropped_width * params->cropped_height) / 
            (params->data_width * params->data_height) * 100);
    
    fclose(meta_file);
    printf("元数据已保存到: %s\n", metadata_filename);
}

// 验证处理结果
void validate_output(ProcessingParams *params) {
    long expected_size = params->output_width * params->output_height * sizeof(ComplexInt16);
    long actual_size = get_file_size(params->output_filename);
    
    if (actual_size != expected_size) {
        fprintf(stderr, "警告: 输出文件大小不匹配\n");
        fprintf(stderr, "预期大小: %ld 字节\n", expected_size);
        fprintf(stderr, "实际大小: %ld 字节\n", actual_size);
    } else {
        printf("输出文件验证通过\n");
    }
}

int main(int argc, char *argv[]) {
    printf("=== SLC多视处理工具 (InSAR专用) ===\n");
    printf("输入输出格式: int16实部 + int16虚部 (4字节/像素)\n\n");
    
    // 解析参数
    ProcessingParams params;
    memset(&params, 0, sizeof(params));
    
    if (!parse_arguments(argc, argv, &params)) {
        return EXIT_FAILURE;
    }
    
    // 计算输出尺寸并裁剪
    calculate_dimensions(&params);
    
    printf("参数设置:\n");
    printf("  输入文件: %s\n", params.input_filename);
    printf("  数据宽度: %d 像素\n", params.data_width);
    printf("  数据高度: %d 像素 (根据文件大小计算)\n", params.data_height);
    printf("  距离向降低: %d 倍\n", params.range_look);
    printf("  方位向降低: %d 倍\n", params.azimuth_look);
    printf("  输出文件: %s\n", params.output_filename);
    printf("\n");
    
    printf("裁剪信息:\n");
    if (params.cropped_width < params.data_width) {
        printf("  宽度裁剪: %d -> %d (丢弃右侧 %d 像素)\n", 
               params.data_width, params.cropped_width, 
               params.data_width - params.cropped_width);
    } else {
        printf("  宽度: 无需裁剪\n");
    }
    
    if (params.cropped_height < params.data_height) {
        printf("  高度裁剪: %d -> %d (丢弃底部 %d 行)\n", 
               params.data_height, params.cropped_height, 
               params.data_height - params.cropped_height);
    } else {
        printf("  高度: 无需裁剪\n");
    }
    printf("\n");
    
    // 执行多视处理
    perform_multi_looking(&params);
    
    // 生成元数据
    generate_metadata_file(&params);
    
    // 验证结果
    validate_output(&params);
    
    printf("\n处理完成！输出文件保持int16复数格式，可用于InSAR处理。\n");
    
    return EXIT_SUCCESS;
}