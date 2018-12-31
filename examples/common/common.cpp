#include <iostream>
#include <fstream>
#include <unistd.h>
#include "tengine_c_api.h"

/*!
* @brief Get the Tengine root path
* @return The string of Tengine root path(e.g. "/home/user/tengine/")
* @note  Firstly, assume the program is under "Tengine_root/build/examples/...";
         if failed, then assume the program is under "Tengine_root/examples/...";
*        if failed, then assume the program is under Tengine_root;
*        if failed again, then return an empty string.
*/
std::string get_root_path(void)
{
    typedef std::string::size_type pos;
    char buf[1024];

    int rslt = readlink("/proc/self/exe", buf, 1023);
    if(rslt < 0 || rslt > 1023)
    {
        return std::string("");
    }
    buf[rslt] = '\0';

    std::string str = buf;
    std::cout << str << std::endl;
    pos p = str.find("build/examples/");
    if(p == std::string::npos)
    {
        p = str.find("examples/");
        if(p == std::string::npos)
        {
            p = str.find("tengine/");
            if(p == std::string::npos)
                return std::string("");
            else
                return str.substr(0, p + 8);
        }
    }
    return str.substr(0, p);
}

/*!
 * @brief Find the config file and set it to tengine
 * @note  Users can export TENGINE_CONFIG_FILE to set the the config file.
 *        If env TENGINE_CONFIG_FILE is not specified,
 *        search the config file in "Tengine_root/install/etc/tengine/config";
 *        if failed again, display error message and return false.
 */
bool set_tengine_config()
{
#if 0  
  const char * env_key="TENGINE_CONFIG_FILE";
    const char * conf_env=std::getenv(env_key);
    std::fstream test_fs;

    // if env TENGINE_CONFIG_FILE is specified ...
    if(conf_env)
    {
        test_fs.open(conf_env);
        if(test_fs.is_open())
        {
            test_fs.close();
            set_config_file(conf_env);
            return true;
        }
        else
        {
            std::cerr << "Can't find tengine config file : " << conf_env << "\n";
            return false;
        }
    }

    // if env TENGINE_CONFIG_FILE is not specified ...
    const std::string config = get_root_path() + "install/etc/tengine/config";
    test_fs.open(config);
    if(test_fs.is_open())
    {
        test_fs.close();
        set_config_file(config.c_str());
        return true;
    }
    else
    {
        std::cerr << "Can't find tengine config file in install dir.\n"
                  << "Please export TENGINE_CONFIG_FILE to set it.\n";
        return false;
    }
#endif
    return true;
}

/*!
 * @brief Find the model file or label file according to the file name
 * @param fname The model file name or label file name
 * @return The fullname of the file that founded
 * @note  Firstly, search the file in current directory;
 *        if failed, search the file in "Tengine_root/models/";
 *        if still failed, display error message and return empty string.
 */
std::string get_file(const char* fname)
{
    std::fstream test_fs;
    std::string fn = fname;

    const std::string mod_sch1 = "./" + fn;
    const std::string mod_sch2 = get_root_path() + "models/" + fn;

    test_fs.open(mod_sch1.c_str());
    if(test_fs.is_open())
    {
        test_fs.close();
        return mod_sch1;
    }
    else
    {
        test_fs.open(mod_sch2.c_str());
        if(test_fs.is_open())
        {
            test_fs.close();
            return mod_sch2;
        }
        else
        {
            std::cerr << "Can't find " << fn << " in current dir and models dir.\n";
            return std::string("");
        }
    }
}

bool check_file_exist(const std::string file_name)
{
    FILE* fp = fopen(file_name.c_str(), "r");
    if(!fp)
    {
        std::cerr << "Input file not existed: " << file_name << "\n";
        return false;
    }
    fclose(fp);
    return true;
}
