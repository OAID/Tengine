#pragma once

#include "node.h"
#include <cassert>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

namespace pipe {
class BaseNode;

class BaseEdge {
public:
  void set_in_node(BaseNode *node, std::string m_name) {
    m_in_node = std::make_pair(node, m_name);
  }

  void set_out_node(BaseNode *node, std::string m_name) {
    m_out_node = std::make_pair(node, m_name);
  }

private:
  std::tuple<BaseNode *, std::string> m_in_node;
  std::tuple<BaseNode *, std::string> m_out_node;
};

template <typename T> class InstantEdge final : public BaseEdge {
public:
  InstantEdge() = delete;

  InstantEdge(size_t cap) {
    assert(cap > 0);
    m_cap = cap;
  }

  bool try_push(const T &val) {
    std::lock_guard<std::mutex> _(m_mtx);
    if (m_queue.size() >= m_cap) {
      return false;
    }
    while (not m_queue.empty()) {
      m_queue.pop();
    }
    m_queue.push(val);
    return true;
  }

  bool try_push(T &&val) {
    std::lock_guard<std::mutex> _(m_mtx);
    if (m_queue.size() >= m_cap) {
      return false;
    }
    while (not m_queue.empty()) {
      m_queue.pop();
    }
    m_queue.push(std::move(val));
    return true;
  }

  bool pop(T &val) {
    std::lock_guard<std::mutex> _(m_mtx);
    if (m_queue.empty()) {
      return false;
    }
    val = m_queue.front();
    m_queue.pop();
    return true;
  }

private:
  std::queue<T> m_queue;
  size_t m_cap = 0;
  std::mutex m_mtx;
};

} // namespace pipe
