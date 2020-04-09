#ifndef __TENGINE_OP_HPP__
#define __TENGINE_OP_HPP__

#include <list>
#include <memory>

namespace tengine
{
    namespace nn
    {
        struct OpData
        {
            OpData()
                :data_(0)
                ,n_(0)
                ,c_(0)
                ,h_(0)
                ,w_(0)
                {}

            OpData(float* dat,int n,int c,int h,int w)
            :data_(dat)
            ,n_(n)
            ,c_(c)
            ,h_(h)
            ,w_(w)
            {}

            float* data_;
            int n_;
            int c_;
            int h_;
            int w_;
        };

        class TengineOp
        {
            public:
                virtual ~TengineOp() {}
                virtual bool run() = 0;
        };

        typedef std::shared_ptr<TengineOp> TTengineOpPtr;
        typedef std::list<TengineOp*> TTengineOpList;

        class TengineOpGraph : public TengineOp
        {
            public:
                TengineOpGraph()
                    {}
                virtual bool add_op(TengineOp* op) = 0;
            protected:
                TTengineOpList _op;
        };

    }
}

#endif
