#pragma once

namespace pipe {
    template <typename T>
    struct Box {
    T x0;
    T y0;
    T x1;
    T y1;
    int class_idx;
    float score;
    };
} // namespace pipe
