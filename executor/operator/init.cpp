namespace TEngine {

extern void NodeOpsRegistryManagerInit(void);
<<<<<<< HEAD
extern void RegisterCommonOps(void);
extern void RegisterRefOps(void);

#if CONFIG_ARCH_ARM64 == 1
extern void RegisterArmOps(void);
#endif

}
=======
extern void RegisterRefOps(void);

#if CONFIG_ARCH_X86
extern void RegisterX86Ops(void);
#endif

#if CONFIG_ARCH_ARM64 == 1 || CONFIG_ARCH_ARM32 == 1
extern void RegisterArmOps(void);
#endif

#if CONFIG_ARCH_ARM8_2 == 1
extern void RegisterArmHalfOps(void);
#endif
}    // namespace TEngine
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

using namespace TEngine;

extern "C" int register_hclcpu_ops(void)
{
<<<<<<< HEAD
    RegisterCommonOps();
    RegisterRefOps();

#if CONFIG_ARCH_ARM64
    RegisterArmOps();
#endif

    return 0;

}

=======
    RegisterRefOps();

#if CONFIG_ARCH_X86
    RegisterX86Ops();
#endif	

#if CONFIG_ARCH_ARM64 || CONFIG_ARCH_ARM32
    RegisterArmOps();
#endif

#if CONFIG_ARCH_ARM8_2
    RegisterArmHalfOps();
#endif

    return 0;
}
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
