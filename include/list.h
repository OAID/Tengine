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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __LIST_H__
#define __LIST_H__

#include "sys_port.h"

#define LIST_DEFINE(name) struct list name = {&(name), &(name)}

struct list
{
    struct list *next, *prev;
};

static inline void init_list(struct list* l)
{
    l->next = l->prev = l;
}

static inline int list_empty(struct list* l)
{
    return (l->next == l->prev);
}

static inline int list_is_last(const struct list* entry, const struct list* l)
{
    return entry->next == l;
}

static inline int list_is_first(const struct list* entry, const struct list* l)
{
    return entry->prev == l;
}

static inline void insert_list_entry(struct list* entry, struct list* prev, struct list* next)
{
    prev->next = entry;
    next->prev = entry;

    entry->next = next;
    entry->prev = prev;
}

/* insert after head node */
static inline void insert_list(struct list* entry, struct list* head)
{
    insert_list_entry(entry, head, head->next);
}

/* insert at the tail */
static inline void append_list(struct list* entry, struct list* head)
{
    insert_list_entry(entry, head->prev, head);
}

static inline void remove_list_entry(struct list* prev, struct list* next)
{
    prev->next = next;
    next->prev = prev;
}

static inline void remove_list(struct list* entry)
{
    remove_list_entry(entry->prev, entry->next);
    entry->prev = NULL;
    entry->next = NULL;
}

static inline void replace_list(struct list* old, struct list* new)
{
    new->next = old->next;
    new->prev = old->prev;

    new->next->prev = new;
    new->prev->next = new;

    init_list(old);
}
#ifdef _MSC_VER
#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_entry_head(list, type, member) list_entry((list)->next, type, member)

#define list_entry_tail(list, type, member) list_entry((list)->prev, type, member)

#define list_entry_next(entry, type, member) list_entry((entry)->member.next, type, member)

#define list_entry_is_last(entry, list, member) list_is_last(&entry->member, list)

#define list_entry_is_first(entry, list, member) list_is_first(&entry->member, list)

#define list_for_each(pos, list) for(pos = (list)->next; pos != (list); pos = pos->next)

#define list_for_each_safe(pos, n, list) for(pos = (list)->next, n = pos->next; pos != (list); pos = n, n = pos->next)

#define list_entry_for_each(pos, type, list, member) \
    for(pos = list_entry_head(list, type, member); &pos->member != (list); pos = list_entry_next(pos, type, member))

#define list_for_each_entry_safe(pos, type, n, list, member)                                                           \
    for(pos = list_entry_head(list, type, member), n = list_entry_next(pos, type, member); &pos->member != (list);     \
        pos = n, n = list_entry_next(n, type, member))
#else
#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_entry_head(list, type, member) list_entry((list)->next, type, member)

#define list_entry_tail(list, type, member) list_entry((list)->prev, type, member)

#define list_entry_next(entry, member) list_entry((entry)->member.next, typeof(*entry), member)

#define list_entry_is_last(entry, list, member) list_is_last(&entry->member, list)

#define list_entry_is_first(entry, list, member) list_is_first(&entry->member, list)

#define list_for_each(pos, list) for(pos = (list)->next; pos != (list); pos = pos->next)

#define list_for_each_safe(pos, n, list) for(pos = (list)->next, n = pos->next; pos != (list); pos = n, n = pos->next)

#define list_entry_for_each(pos, list, member) \
    for(pos = list_entry_head(list, typeof(*pos), member); &pos->member != (list); pos = list_entry_next(pos, member))

#define list_for_each_entry_safe(pos, n, list, member)                                                               \
    for(pos = list_entry_head(list, typeof(*pos), member), n = list_entry_next(pos, member); &pos->member != (list); \
        pos = n, n = list_entry_next(n, member))

#endif

#endif
