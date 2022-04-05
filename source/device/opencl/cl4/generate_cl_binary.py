import os


def convert_string_to_hex_list(code_str):
    hex_list = []
    for i in range(len(code_str)):
        hex_temp = hex(ord(code_str[i]))
        if hex_temp == '0x5411':
            print(code_str[i])
        hex_list.append(hex_temp)
    return hex_list


def opencl_codegen():
    cl_kernel_dir = "./"
    output_path = "./ocl_program_hex.cc"
    opencl_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
                code_str = ""
                for line in f.readlines():
                    code_str += line
                opencl_code_maps[file_name[:-3]] = convert_string_to_hex_list(code_str)

    opencl_source_map = "#include <map>\n"
    opencl_source_map += "#include <string>\n"
    opencl_source_map += "#include <vector>\n"
    opencl_source_map += "extern const std::map<std::string, std::vector<unsigned char>> opencl_program_map = \n { \n"
    items = opencl_code_maps.items()
    for file_name, file_source in items:
        opencl_source_map += "{\n \""
        opencl_source_map += file_name
        opencl_source_map += "\", \n"
        opencl_source_map += "     { "
        for source_hex in file_source:
            opencl_source_map += source_hex
            opencl_source_map += ","
        opencl_source_map += " } "
        opencl_source_map += "\n }, \n"

    opencl_source_map += " }; \n"

    with open(output_path, "w") as w_file:
        w_file.write(opencl_source_map)


if __name__ == '__main__':
    opencl_codegen()
