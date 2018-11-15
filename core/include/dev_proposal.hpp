#ifndef __DEV_PROPOSAL_HPP__
#define __DEV_PROPOSAL_HPP__

#include <string>

namespace TEngine {

#define DEV_PROPOSAL_ATTR "dev_proposal"

#define DEV_PROPOSAL_UNSPPORT 0
#define DEV_PROPOSAL_CAN_DO   1
#define DEV_PROPOSAL_GOODAT   2
#define DEV_PROPOSAL_BEST     3
#define DEV_PROPOSAL_STATIC   4

struct DevProposal
{
       std::string dev_id;
       int level;  
  
       DevProposal(): level(DEV_PROPOSAL_UNSPPORT){};
};




} //namespace TEngine



#endif
