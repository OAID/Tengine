INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include
INC_DIR+=-I$(shell pwd)/../include
INC_DIR+=-I$(shell pwd)/include

MODULE_DIR+=

obj-y+=tengine/
COMMON_CFLAGS+= -DCONFIG_TENGINE_SERIALIZER

CXXFLAGS+=
COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) -Werror

obj-y+=plugin/

install:



