BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/include 
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../core/include/olreport
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../serializer/include
INC_DIR+=-I$(shell pwd)/../driver/include
INC_DIR+=-I$(shell pwd)/../executor/include

CXXFLAGS+=

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g  -I$(shell pwd)/include -fPIC  $(INC_DIR) $(PROTOBUF_CFLAGS)

binsubdir_list=./bin


include ./bin/Makefile

INTERNAL_DIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/,$(bin-obj-y))
bin-obj-y:=



BIN_OBJS=$(INTERNAL_DIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)


default: $(binsubdir_list)  $(BIN_EXES)

test: $(BIN_EXES)


$(BIN_OBJS): $(binsubdir_list) ;

build: default install


install:
	@mkdir -p $(INSTALL_DIR)/bin
	@cp -f $(BIN_EXES) $(INSTALL_DIR)/bin

$(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

clean: $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default test install  $(binsubdir_list)


$(BUILD_DIR)/%: $(BUILD_DIR)/%.o 
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS)

LIBS+=-L../build -ltengine -Wl,--allow-shlib-undefined

LIBS+=$(PROTOBUF_LIB)
LIBS+=$(OPENBLAS_LIB)
LIBS+= -ldl -lpthread -lresolv 
SO_LIBS+=-Wl,-rpath,./build/
