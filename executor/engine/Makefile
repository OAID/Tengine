obj-y+=simple_executor.o
obj-$(CONFIG_EVENT_EXECUTOR)+=event.o
obj-$(CONFIG_EVENT_EXECUTOR)+=event_executor.o

simple_executor_CXXFLAGS+=-O2

EVENT_CXXFLAGS+=-I ../include/event
EVENT_CXXFLAGS+=-I ../../devices/hal/include
event_CXXFLAGS+=$(EVENT_CXXFLAGS)
event_executor_CXXFLAGS+=$(EVENT_CXXFLAGS)
