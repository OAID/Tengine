/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __TENSOR_SHAPE_HPP__
#define __TENSOR_SHAPE_HPP__

#include <vector>
#include <string>
#include <iostream>


namespace TEngine {

enum  TensorType {
   kVarTensor,
   kConstTensor,
   kInputTensor,
   kDepTensor
};

class TShape {

public:
	void SetDataLayout(const std::string& layout_name)
	{
               layout_=layout_name;
	}

	const std::string& GetDataLayout(void) const
	{
		return layout_;
	}

	int GetSize(void) const 
        { 
            if(dim_.size()==0)
                 return 0;

            int result=1;
          
            for(unsigned int i=0;i<dim_.size();i++)
                result*=dim_[i];
            
            return result;
        }

        std::vector<int>& GetDim(void) { return dim_;}

        const std::vector<int>& GetDim(void) const { return dim_;}


        int Shape(unsigned int idx) const
        {
            if(idx<dim_.size())
               return dim_[idx];

            return 1;
        }

        bool SetShape(unsigned int idx, int val)
        {
            if(idx<dim_.size())
                return false;
            
             dim_[idx]=val;

             return true;
        }

	void SetDim(const std::vector<int>& args, bool layout_check=false);


        void DumpShape(std::ostream& os) const;

        int GetN(void) const;
        int GetC(void) const;
        int GetH(void) const;
        int GetW(void) const;
        int GetD(void) const;

        TShape()=default;


	TShape(const TShape& src) 
	{ 
		dim_=src.dim_; 
		layout_=src.layout_;
	}

	TShape(TShape&& src)
	{
		dim_=std::move(src.dim_);
		layout_=src.layout_;
	}


	TShape& operator=(const TShape& rhs) 
	{
		dim_=rhs.dim_;
		layout_=rhs.layout_;

		return *this;
	}

	TShape& operator=(TShape&& rhs)
	{
		dim_=std::move(rhs.dim_);
		layout_=rhs.layout_;

		return *this;
	}

        bool operator==(const TShape & rhs)
        {
             if(layout_==rhs.layout_ &&
                 dim_==rhs.dim_)

               return true;
            else
               return false;
        }



	virtual ~TShape(){};


private:

	std::vector<int> dim_;
	std::string layout_;

};


} //namesapce TEngine



#endif


