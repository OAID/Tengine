#ifndef __ATTR_IO_H__
#define __ATTR_IO_H__

namespace TEngine {

struct AttrIO {

	using get_func_t=std::function<bool(const char * name, void *, int)>;
	using set_func_t=std::function<bool(const char * name, const void*, int)>;


	std::unordered_map<std::string, get_func_t> get_func_map;
	std::unordered_map<std::string, set_func_t> set_func_map;

	get_func_t bailout_get; //for not registered attr name
	set_func_t bailout_set; //for not registered attr name



bool SetAttr(const char * name, const void * val, int size)
{
	if(set_func_map.count(name)==0)
	{
		if(bailout_set==nullptr)
			return false;

		return bailout_set(name,val,size);
	}

	set_func_t set_func=set_func_map.at(name);

	return set_func(name,val,size);
}

bool GetAttr(const char * name,void * val, int size)
{
	if(get_func_map.count(name)==0)
	{
		if(bailout_get==nullptr)
			return false;

		return bailout_get(name,val,size);
	}

	get_func_t get_func=get_func_map.at(name);

	return get_func(name,val,size);
}

bool RegGetFunc(const char *name, get_func_t func)
{
	if(name==nullptr)
	{
		bailout_get=func;
		return true;
	}

	get_func_map[name]=func;

	return true;
}

bool RegSetFunc(const char * name, set_func_t func)
{
	if(name==nullptr)
	{
		bailout_set=func;
		return true;
	}

	set_func_map[name]=func;

	return true;

}

};

} //namespace TEngine


#endif


