/**
 * \file sdk/load-and-run/src/text_table.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <array>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace pipeline {

class TextTable
{
public:
    enum Level
    {
        Summary,
        Detail
    };
    enum class Align : int
    {
        Left,
        Right,
        Mid
    };
    TextTable() = default;
    explicit TextTable(const std::string& table_name)
        : m_name(table_name)
    {
    }
    TextTable& horizontal(char c)
    {
        m_row.params.horizontal = c;
        return *this;
    }
    TextTable& vertical(char c)
    {
        m_row.params.vertical = c;
        return *this;
    }
    TextTable& corner(char c)
    {
        m_row.params.corner = c;
        return *this;
    }
    TextTable& align(Align v)
    {
        m_row.params.align = v;
        return *this;
    }
    TextTable& padding(size_t w)
    {
        m_padding = w;
        return *this;
    }
    TextTable& prefix(const std::string& str)
    {
        m_prefix = str;
        return *this;
    }

    template<typename T>
    TextTable& add(const T& value)
    {
        m_row.values.emplace_back(value);
        if (m_cols_max_w.size() < m_row.values.size())
        {
            m_cols_max_w.emplace_back(m_row.values.back().length());
        }
        else
        {
            size_t i = m_row.values.size() - 1;
            m_cols_max_w[i] = std::max(m_cols_max_w[i], m_row.values.back().length());
        }
        return *this;
    }

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = 0>
    TextTable& add(const T& value)
    {
        std::stringstream ss;
        ss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        ss << value;
        m_row.values.emplace_back(ss.str());
        if (m_cols_max_w.size() < m_row.values.size())
        {
            m_cols_max_w.emplace_back(m_row.values.back().length());
        }
        else
        {
            size_t i = m_row.values.size() - 1;
            m_cols_max_w[i] = std::max(m_cols_max_w[i], m_row.values.back().length());
        }
        return *this;
    }

    template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = 0>
    TextTable& add(const T& value)
    {
        m_row.values.emplace_back(std::to_string(value));
        return *this;
    }

    void eor()
    {
        m_rows.emplace_back(m_row);
        adjuster_last_row();
        m_row.values.clear();
    }

    void reset()
    {
        m_row = {};
        m_cols_max_w.clear();
        m_padding = 0;
        m_rows.clear();
    }

    void show(std::ostream& os)
    {
        if (m_rows.empty()) return;
        auto& last_row = m_rows.front();
        bool first = true;
        for (auto& row : m_rows)
        {
            auto& lrow = (last_row.values.size() * char_length(last_row.params.horizontal)) > (row.values.size() * char_length(row.params.horizontal))
                             ? last_row
                             : row;
            // line before row
            if (lrow.params.horizontal)
            {
                if (not first) os << std::endl;
                os << m_prefix;
                if (lrow.params.corner) os << lrow.params.corner;
                size_t skip_size = 0;
                // table name
                if (first)
                {
                    os << m_name;
                    skip_size = m_name.length();
                }
                for (size_t i = 0; i < lrow.values.size(); ++i)
                {
                    auto max_w = m_cols_max_w.at(i) + m_padding * 2;
                    if (max_w + char_length(lrow.params.corner) <= skip_size)
                    {
                        skip_size = skip_size - max_w - char_length(lrow.params.corner);
                        continue;
                    }
                    size_t rest = max_w + char_length(lrow.params.corner) - skip_size;
                    skip_size = 0;
                    if (rest > char_length(lrow.params.corner))
                    {
                        os << std::string(rest - char_length(lrow.params.corner),
                                          lrow.params.horizontal);
                        rest = char_length(lrow.params.corner);
                    }
                    if (rest > 0 && lrow.params.corner) os << lrow.params.corner;
                }
            }
            else if (first)
            {
                os << m_prefix << ' ' << m_name;
            }
            first = false;
            os << std::endl
               << m_prefix;
            if (row.params.vertical) os << row.params.vertical;
            // row
            for (size_t i = 0; i < row.values.size(); ++i)
            {
                auto& str = row.values.at(i);
                auto max_w = m_cols_max_w.at(i) + 2 * m_padding;
                if (row.params.align == Align::Mid)
                {
                    mid(os, str, max_w);
                }
                else if (row.params.align == Align::Left)
                {
                    os << std::setw(max_w) << std::left << str;
                }
                else
                {
                    os << std::setw(max_w) << std::right << str;
                }
                if (row.params.vertical) os << row.params.vertical;
            }
            last_row = row;
        }
        if (last_row.params.horizontal)
        {
            os << std::endl
               << m_prefix;
            if (last_row.params.corner) os << last_row.params.corner;
            for (size_t i = 0; i < last_row.values.size(); ++i)
            {
                auto max_w = m_cols_max_w.at(i);
                std::string tmp(max_w + m_padding * 2, last_row.params.horizontal);
                os << tmp;
                if (last_row.params.corner) os << last_row.params.corner;
            }
        }
    }

private:
    void adjuster_last_row()
    {
        if (m_rows.empty()) return;
        auto& row = m_rows.back();
        if (row.params.horizontal == 0 or row.params.vertical == 0)
        {
            row.params.corner = 0;
        }
        if (row.params.horizontal != 0 && row.params.vertical != 0 && row.params.corner == 0)
        {
            row.params.corner = row.params.horizontal;
        }
    }

    inline void mid(std::ostream& os, const std::string& str, size_t max_w)
    {
        size_t l = (max_w - str.length()) / 2 + str.length();
        size_t r = max_w - l;
        os << std::setw(l) << std::right << str;
        if (r > 0) os << std::setw(r) << ' ';
    }
    inline size_t char_length(char c)
    {
        return c ? 1 : 0;
    }
    std::string m_name;
    std::vector<size_t> m_cols_max_w;
    size_t m_padding = 0;
    std::string m_prefix = "";
    struct Row
    {
        std::vector<std::string> values;
        struct Params
        {
            Align align = Align::Left;
            char horizontal = '-', vertical = '|', corner = '+';
        } params;
    };
    std::vector<Row> m_rows;
    Row m_row;
};

inline std::ostream& operator<<(std::ostream& stream, TextTable& table)
{
    table.show(stream);
    return stream;
}

} // namespace pipeline