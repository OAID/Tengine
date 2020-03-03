#include "thread.h"

typedef struct
{
    THREAD_CONTEXT ctx_;
    int self_data_;
}CUR_THREAD_CONTEXT;

void* worker_run(void *ctx)
{
    CUR_THREAD_CONTEXT* context = (CUR_THREAD_CONTEXT*)ctx;
    printf("current data : %d\n",context->self_data_);

    for(int ii=0; ii<1000; ++ii)
     {
        printf("%d\n",ii);
     }

     return NULL;
}

int main()
{
     CUR_THREAD_CONTEXT ctx;
     init_thread_context(&ctx.ctx_);
     ctx.ctx_.runner_ = worker_run;
     start_thread(&ctx);
     for(int ii=0; ii<1000; ++ii)
     {
        ctx.self_data_ ++;               
     }

     //stop_thread(&ctx);
     join_thread(&(ctx.ctx_));
     return 0;
}
