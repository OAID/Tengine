#include "clientsocket.h"
#include <stdio.h>

int main()
{
    THttpReq req;

    init_post_req(&req,"192.168.93.77",5000,"/testTengine");
    
    char test_data[1024];

    int len = sprintf(test_data,"{\"test\":1111111111111111}");

    char datlen[10] = {'\0'};
    sprintf(datlen," %d",len);
    add_header(&req,"Content-Type","application/json");
    add_header(&req,"Accept","text/plain");

    add_header(&req,"Content-Length",datlen);

    set_data(&req,test_data,len);

    http_post(&req);

    printf("res status : %d\n",req.status);

    return 0;
}
