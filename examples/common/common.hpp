#ifndef __COMMON_HPP__
#define __COMMON_HPP__

std::string get_root_path(void);
bool set_tengine_config(void);
std::string get_file(const char* fname);
bool check_file_exist(const std::string file_name);

#endif    // __COMMON_HPP__
