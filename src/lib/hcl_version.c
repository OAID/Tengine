#include <stdio.h>

#define HCL_VERSION "1.0"

char* get_hcl_version(void)
{
    static char hcl_version[64];
    const char* postfix = "dev";

#ifdef CONFIG_INTERN_RELEASE
    postfix = "release";
#endif

#ifdef CONFIG_INTERN_TRIAL
    postfix = "trial";
#endif

#ifdef CONFIG_AUTHENICATION
    postfix = "authed";
#endif
    int ret = snprintf(hcl_version, 64, "%s-%s", HCL_VERSION, postfix);

    if (ret >= 64)
        hcl_version[63] = 0;
    return hcl_version;
}
