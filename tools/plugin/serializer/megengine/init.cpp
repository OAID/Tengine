
#include <iostream>
#include <functional>

#include "logger.hpp"
#include "megengine_serializer.hpp"

namespace TEngine {
extern bool MegengineSerializerRegisterOpLoader();
}    // namespace TEngine

using namespace TEngine;

extern "C" int megengine_plugin_init(void)
{
    auto factory = SerializerFactory::GetFactory();

    factory->RegisterInterface<MegengineSerializer>("megengine");
    auto megengine_serializer = factory->Create("megengine");

    SerializerManager::SafeAdd("megengine", SerializerPtr(megengine_serializer));
    MegengineSerializerRegisterOpLoader();

    return 0;
}
