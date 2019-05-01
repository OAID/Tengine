namespace TEngine {

extern void NodeOpsRegistryManagerInit(void);
extern void RegisterCommonOps(void);
extern void RegisterRefOps(void);

#if CONFIG_ARCH_ARM64 == 1
extern void RegisterArmOps(void);
#endif

}

using namespace TEngine;

extern "C" int register_hclcpu_ops(void)
{
    RegisterCommonOps();
    RegisterRefOps();

#if CONFIG_ARCH_ARM64
    RegisterArmOps();
#endif

    return 0;

}

