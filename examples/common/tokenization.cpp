#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include "utf8proc.h"

#include "./tokenization.h"


namespace cuBERT {

    void FullTokenizer::convert_tokens_to_ids(const std::vector<std::string> &tokens, uint64_t *ids) {
        for (int i = 0; i < tokens.size(); ++i) {
            ids[i] = convert_token_to_id(tokens[i]);
        }
    }

// trim from start (in place)
    static inline void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        }));
    }

// trim from end (in place)
    static inline void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

// trim from both ends (in place)
    static inline void trim(std::string &s) {
        ltrim(s);
        rtrim(s);
    }

    void load_vocab(const char *vocab_file, std::unordered_map<std::string, uint64_t> *vocab) {
        std::ifstream file(vocab_file);
        if (!file) {
            throw std::invalid_argument("Unable to open vocab file");
        }

        unsigned int index = 0;
        std::string line;
        while (std::getline(file, line)) {
            trim(line);
            (*vocab)[line] = index;
            index++;
        }

        file.close();
    }

    inline bool _is_whitespace(int c, const char *cat) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }
        return cat[0] == 'Z' && cat[1] == 's';
    }

    inline bool _is_control(int c, const char *cat) {
        // These are technically control characters but we count them as whitespace characters.
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        return 'C' == *cat;
    }

    inline bool _is_punctuation(int cp, const char *cat) {
// We treat all non-letter/number ASCII as punctuation.
// Characters such as "^", "$", and "`" are not in the Unicode
// Punctuation class but we treat them as punctuation anyways, for
// consistency.
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        return 'P' == *cat;
    }

    bool _is_whitespace(int c) {
        return _is_whitespace(c, utf8proc_category_string(c));
    }

    bool _is_control(int c) {
        return _is_control(c, utf8proc_category_string(c));
    }

    bool _is_punctuation(int cp) {
        return _is_punctuation(cp, utf8proc_category_string(cp));
    }

    bool BasicTokenizer::_is_chinese_char(int cp) {
// This defines a "chinese character" as anything in the CJK Unicode block:
//   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
//
// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
// despite its name. The modern Korean Hangul alphabet is a different block,
// as is Japanese Hiragana and Katakana. Those alphabets are used to write
// space-separated words, so they are not treated specially and handled
// like the all of the other languages.
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF) ||
               (cp >= 0x2A700 && cp <= 0x2B73F) ||
               (cp >= 0x2B740 && cp <= 0x2B81F) ||
               (cp >= 0x2B820 && cp <= 0x2CEAF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x2F800 && cp <= 0x2FA1F);
    }

    void BasicTokenizer::tokenize(const char *text, std::vector<std::string> *output_tokens, size_t max_length) {
// This was added on November 1st, 2018 for the multilingual and Chinese
// models. This is also applied to the English models now, but it doesn't
// matter since the English models were not trained on any Chinese data
// and generally don't have any Chinese data in them (there are Chinese
// characters in the vocabulary because Wikipedia does have some Chinese
// words in the English Wikipedia.).
        if (do_lower_case) {
            text = (const char *) utf8proc_NFD((const utf8proc_uint8_t *) text);
        }

        size_t word_bytes = std::strlen(text);
        bool new_token = true;
        size_t subpos = 0;
        int cp;
        char dst[4];

        while (word_bytes > 0) {
            int len = utf8proc_iterate((const utf8proc_uint8_t *) text + subpos, word_bytes, &cp);
            if (len < 0) {
                std::cerr << "UTF-8 decode error: " << text << std::endl;
                break;
            }
            if (do_lower_case) {
                cp = utf8proc_tolower(cp);
            }

            const char *cat = utf8proc_category_string(cp);
            if (cp == 0 || cp == 0xfffd || _is_control(cp, cat)) {
                // pass
            } else if (do_lower_case && cat[0] == 'M' && cat[1] == 'n') {
                // pass
            } else if (_is_whitespace(cp, cat)) {
                new_token = true;
            } else {
                size_t dst_len = len;
                const char *dst_ptr = text + subpos;
                if (do_lower_case) {
                    dst_len = utf8proc_encode_char(cp, (utf8proc_uint8_t *) dst);
                    dst_ptr = dst;
                }

                if (_is_punctuation(cp, cat) || _is_chinese_char(cp)) {
                    output_tokens->emplace_back(dst_ptr, dst_len);
                    new_token = true;
                } else {
                    if (new_token) {
                        output_tokens->emplace_back(dst_ptr, dst_len);
                        new_token = false;
                    } else {
                        output_tokens->at(output_tokens->size() - 1).append(dst_ptr, dst_len);
                    }
                }
            }

            word_bytes = word_bytes - len;
            subpos = subpos + len;

            // early terminate
            if (output_tokens->size() >= max_length) {
                break;
            }
        }

        if (do_lower_case) {
            free((void *) text);
        }
    }


    void WordpieceTokenizer::tokenize(const std::string &token, std::vector<std::string> *output_tokens) {
        if (token.size() > max_input_chars_per_word) {  // FIXME: slightly different
            output_tokens->push_back(unk_token);
            return;
        }
        size_t output_tokens_len = output_tokens->size();

        for (size_t start = 0; start < token.size();) {
            bool is_bad = true;

            // TODO: can be optimized by prefix-tree
            for (size_t end = token.size(); start < end; --end) {  // FIXME: slightly different
                std::string substr = start > 0
                                     ? "##" + token.substr(start, end - start)
                                     : token.substr(start, end - start);
                if (vocab->count(substr)) {
                    is_bad = false;
                    output_tokens->push_back(substr);
                    start = end;
                    break;
                }
            }

            if (is_bad) {
                output_tokens->resize(output_tokens_len);
                output_tokens->push_back(unk_token);
                return;
            }
        }
    }


    void FullTokenizer::tokenize(const char *text, std::vector<std::string> *output_tokens, size_t max_length) {
        std::vector<std::string> tokens;
        tokens.reserve(max_length);
        basic_tokenizer->tokenize(text, &tokens, max_length);

        for (const auto &token : tokens) {
            wordpiece_tokenizer->tokenize(token, output_tokens);

            // early terminate
            if (output_tokens->size() >= max_length) {
                break;
            }
        }
    }
}
