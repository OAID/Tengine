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
#ifndef __DATA_LAYOUT_HPP__
#define __DATA_LAYOUT_HPP__

#include "named_data.hpp"

namespace TEngine {

struct  DataLayout: public NamedData<DataLayout> {

	DataLayout(const std::string& str, bool as_default=false)
	{
		layout_name=str;

		SetData(layout_name,this);

                if(as_default)
                   SetDefaultData(this);
	}

	DataLayout(std::string&& str,bool as_default=false)
	{
		layout_name=std::move(str);
		SetData(layout_name,this);

                if(as_default)
                   SetDefaultData(this);
	}

	static  const DataLayout * GetLayout( const std::string& name)
	{
		return GetData(name);
	}

	const std::string& GetName(void) const { return layout_name;}

        virtual unsigned int GetDimNum() const {return 0;}
	virtual int GetH() const {return -1;} 
	virtual int GetW() const {return -1;} 
	virtual int GetC() const {return -1;} 
	virtual int GetD() const {return -1;}
	virtual int GetN() const {return -1;}

	virtual ~DataLayout(){};

	std::string layout_name;
};


struct LayoutNCHW: public DataLayout {

	LayoutNCHW(bool as_default=false):DataLayout("NCHW",as_default){};

	int GetN() const { return 0;}
	int GetC() const { return 1;}
	int GetH() const { return 2;}
	int GetW() const { return 3;}
        unsigned int GetDimNum() const { return 4;}
};


struct LayoutNCDHW: public DataLayout {

	LayoutNCDHW(bool as_default=false):DataLayout("NCDHW",as_default){};

	int GetN() const { return 0;}
	int GetC() const { return 1;}
	int GetD() const { return 2;}
	int GetH() const { return 3;}
	int GetW() const { return 4;}
        unsigned int GetDimNum() const { return 5;}
};


struct LayoutNHWC: public DataLayout {

	LayoutNHWC(bool as_default=false):DataLayout("NHWC",as_default){};

	int GetN() const { return 0;}
	int GetH() const { return 1;}
	int GetW() const { return 2;}
	int GetC() const { return 3;}
        unsigned int GetDimNum() const { return 4;}
};


struct LayoutNDHWC: public DataLayout {

        LayoutNDHWC(bool as_default=false):DataLayout("NDHWC",as_default){};

        int GetN() const { return 0;}
        int GetD() const { return 1;}
        int GetH() const { return 2;}
        int GetW() const { return 3;}
        int GetC() const { return 4;}
        unsigned int GetDimNum() const { return 5;}
};


struct LayoutNHW: public DataLayout {

	LayoutNHW(bool as_default=false):DataLayout("NHW",as_default){};

	int GetN() const { return 0;}
	int GetH() const { return 1;}
	int GetW() const { return 2;}
        unsigned int GetDimNum() const { return 3;}
};

struct LayoutNW: public DataLayout {

	LayoutNW(bool as_default=false):DataLayout("NW",as_default){};

	int GetN() const { return 0;}
	int GetW() const { return 1;}
        unsigned int GetDimNum() const { return 2;}
};

struct LayoutHW: public DataLayout {

        LayoutHW(bool as_default=false):DataLayout("HW",as_default){};

        int GetH() const { return 0;}
        int GetW() const { return 1;}
        unsigned int GetDimNum() const { return 2;}
};

struct LayoutW: public DataLayout {

	LayoutW(bool as_default=false):DataLayout("W",as_default){};

	int GetW() const { return 0;}
        unsigned int GetDimNum() const { return 1;}
};

} //namespace TEngine

#endif
