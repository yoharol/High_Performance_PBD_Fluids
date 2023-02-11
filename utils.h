#ifndef UTILS_H_
#define UTILS_H_

#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>
#include <memory>
#include <random>
#include <iomanip>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace utils {
template <typename T>
inline T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
inline T max(T a, T b) {
  return a > b ? a : b;
}

inline float random_uniform() {
  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}

class Timer {
 public:
  void setTimer() {
    if (m_base_timer != nullptr) {
      std::cerr << "Warning! Overwriting existing timer\n";
      m_base_timer = nullptr;
    }
    m_base_timer = std::make_unique<BaseTimer>(
        BaseTimer{std::chrono::system_clock::now()});
  }

  double getTimer(std::string message = "", double prevtime = -1.0) {
    if (m_base_timer == nullptr) {
      std::cerr << "Error! No timer initialized\n";
      return -1.0;
    } else {
      const auto time_in_microsec = get_elapsed_time();
      const auto time_in_millisec = time_in_microsec / 1000.0;
      auto time_in_second = time_in_millisec / 1000.0;
      if (prevtime != -1.0f) time_in_second -= prevtime;
      if (message.length() > 0) {
        std::ostringstream sstream;
        sstream << "[ " << std::setw(max<int>(30, message.length())) << message
                << " :\t";
        if (prevtime != -1.0 || time_in_second > 1.0)
          sstream << std::setw(8) << time_in_second << "s \t]" << std::endl;
        else
          sstream << std::setw(8) << time_in_millisec << "ms\t]" << std::endl;
        std::cout << sstream.str() << std::flush;
      }
      return time_in_second;
    }
  }

  double endTimer(std::string message = "") {
    double t = getTimer(message);
    m_base_timer = nullptr;
    return t;
  }

  int64_t get_elapsed_time() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now() - m_base_timer->m_start_time)
        .count();
  }

 private:
  typedef struct {
    std::chrono::system_clock::time_point m_start_time;
  } BaseTimer;

  std::unique_ptr<BaseTimer> m_base_timer = nullptr;
};

class GreyPNGWriter {
 private:
  std::vector<unsigned char> image_buffer;
  int width;
  int height;

 public:
  GreyPNGWriter(int width, int height) {
    this->image_buffer = std::vector<unsigned char>(width * height);
    this->width = width;
    this->height = height;
  }

  void clearBuffer() {
    std::memset(this->image_buffer.data(), 0,
                sizeof(unsigned char) * this->width * this->height);
  }

  void drawOnPos(float x, float y, unsigned char a) {
    int posx = x * width;
    int posy = (1.0 - y) * height;
    this->image_buffer[posy * this->width + posx] = a;
  }

  void drawOnPixel(int posx, int posy, unsigned char a) {
    this->image_buffer[(this->height - posy) * this->width + posx] = a;
  }

  void writeToFile(const char* filename) {
    stbi_write_png(filename, this->width, this->height, 1, image_buffer.data(),
                   width * sizeof(unsigned char));
  }
};

template <typename T>
void writeArrayToFile(std::string filename, T* array, int len) {
  std::ofstream outputfile;
  outputfile.open(filename);
  for (int i = 0; i < len; i++) outputfile << array[i] << std::endl;
  outputfile.close();
}

template <typename T>
T getArrayAverage(T* array, int start, int end) {
  T array_sum = array[start];
  for (int i = start + 1; i < end; i++) array_sum += array[i];
  return array_sum / (end - start);
}

template <typename T>
T getMaxValue(T* array, int start, int end) {
  T array_max = array[start];
  for (int i = start + 1; i < end; i++) array_max = max(array_max, array[i]);
  return array_max;
}

}  // namespace utils
#endif  // UTILS_H_
