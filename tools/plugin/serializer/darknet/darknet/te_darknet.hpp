#ifndef __TE_DARKNET_HPP__
#define __TE_DARKNET_HPP__

typedef struct node
{
    void *val;
    struct node* next;
    struct node* prev;
}node;   

typedef struct list
{
    int size;
    node *front;
    node *back;
} list;

typedef struct
{
    char *type;
    list *options;
}section;   

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

inline LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;

    return BLANK;
}

inline char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = (char*)malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)realloc(line, size*sizeof(char));
            if(!line) {
                //printf("%ld\n", (long)size);
                //malloc_error();
            }
        }
        size_t readsize = size-curr;
        //if(readsize > INT_MAX) readsize = INT_MAX-1;
        if(!fgets(&line[curr], readsize, fp))
        {
            free(line);
            return 0;
        }
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}
inline void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}
inline void remove_str(std::string &src,std::string& str)
{
    size_t len = str.length();
    while(true)
    {
        std::string::size_type pos(0);
        if((pos=src.find(str))!=std::string::npos)
            src.erase(pos,len);
        else
            break;
    }
}
inline void replace_str(std::string &s,std::string& src, std::string& dst)
{
    while(true)
    {
        std::string::size_type pos(0);
        if((pos=s.find(src))!=std::string::npos)
            s.replace(pos,src.length(),dst);
        else
            break;
    }
}
inline void list_insert(list *l, void *val)
{
    node *new_node = (node*)malloc(sizeof(node));
    new_node->val = val;
    new_node->next = 0;

    if(!l->back){
        l->front = new_node;
        new_node->prev = 0;
    }else{
        l->back->next = new_node;
        new_node->prev = l->back;
    }
    l->back = new_node;
    ++l->size;
}
inline list* make_list()
{
    list *l = (list*)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

inline void option_insert(list *l, char *key, char *val)
{
    kvp *p = (kvp*)malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

inline int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}
inline void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}
inline void free_node(node *n)
{
    node *next;
    while(n) {
        next = n->next;
        free(n);
        n = next;
    }
}

inline void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

inline list* read_cfg(const char* filename)
{
    FILE *file = fopen(filename, "r");
    //if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section*)malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
   // return nullptr;
}
inline char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
inline char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

inline int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

inline int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

inline float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

inline float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
#endif 
