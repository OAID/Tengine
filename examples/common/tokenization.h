#ifndef CUBERT_TOKENIZATION_H
#define CUBERT_TOKENIZATION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace cuBERT {

    void load_vocab(const char *vocab_file, std::unordered_map<std::string, uint64_t> *vocab);

/**
 * Checks whether `chars` is a whitespace character.
 * @param c
 * @return
 */
    bool _is_whitespace(int c);

/**
 * Checks whether `chars` is a control character.
 * @param c
 * @return
 */
    bool _is_control(int c);

/**
 * Checks whether `chars` is a punctuation character.
 * @param cp
 * @return
 */
    bool _is_punctuation(int cp);

/**
 * Runs basic tokenization (punctuation splitting, lower casing, etc.).
 */
    class BasicTokenizer {
    public:
        /**
         * Constructs a BasicTokenizer.
         * @param do_lower_case Whether to lower case the input.
         */
        explicit BasicTokenizer(bool do_lower_case = true) : do_lower_case(do_lower_case) {}

        BasicTokenizer(const BasicTokenizer &other) = delete;

        virtual ~BasicTokenizer() = default;

        /**
         * Tokenizes a piece of text.
         *
         * to_lower
         * _run_strip_accents Strips accents from a piece of text.
         * _clean_text Performs invalid character removal and whitespace cleanup on text.
         * _tokenize_chinese_chars Adds whitespace around any CJK character.
         * _run_split_on_punc Splits punctuation on a piece of text.
         * whitespace_tokenize Runs basic whitespace cleaning and splitting on a piece of text.
         *
         * @param text
         * @param output_tokens
         */
        void tokenize(const char *text, std::vector<std::string> *output_tokens, size_t max_length);

    private:
        const bool do_lower_case;

        /**
         * Checks whether CP is the codepoint of a CJK character.
         * @param cp
         * @return
         */
        inline static bool _is_chinese_char(int cp);
    };

/**
 * Runs WordPiece tokenziation.
 */
    class WordpieceTokenizer {
    public:
        explicit WordpieceTokenizer(
                std::unordered_map<std::string, uint64_t> *vocab,
                std::string unk_token = "[UNK]",
                int max_input_chars_per_word = 200
        ) : vocab(vocab), unk_token(unk_token), max_input_chars_per_word(max_input_chars_per_word) {}

        WordpieceTokenizer(const WordpieceTokenizer &other) = delete;

        virtual ~WordpieceTokenizer() = default;

        /**
         * Tokenizes a piece of text into its word pieces.
         *
         * This uses a greedy longest-match-first algorithm to perform tokenization
         * using the given vocabulary.
         *
         * For example:
         *   input = "unaffable"
         *   output = ["un", "##aff", "##able"]
         *
         * @param text A single token or whitespace separated tokens. This should have already been passed through `BasicTokenizer.
         * @param output_tokens A list of wordpiece tokens.
         */
        void tokenize(const std::string &text, std::vector<std::string> *output_tokens);

    private:
        const std::unordered_map<std::string, uint64_t> *vocab;
        const std::string unk_token;
        const int max_input_chars_per_word;
    };


/**
 * Runs end-to-end tokenziation.
 */
    class FullTokenizer {
    public:
        explicit FullTokenizer(const char *vocab_file, bool do_lower_case = true) {
            vocab = new std::unordered_map<std::string, uint64_t>();
            load_vocab(vocab_file, vocab);

            basic_tokenizer = new BasicTokenizer(do_lower_case);
            wordpiece_tokenizer = new WordpieceTokenizer(vocab);
        }

        FullTokenizer(const FullTokenizer &other) = delete;

        virtual ~FullTokenizer() {
            delete wordpiece_tokenizer;
            delete basic_tokenizer;
            delete vocab;
        }

        void tokenize(const char *text, std::vector<std::string> *output_tokens, size_t max_length);

        inline uint64_t convert_token_to_id(const std::string &token) {
            auto item = vocab->find(token);
            if (item == vocab->end()) {
                std::cerr << "vocab missing key: " << token << std::endl;
                return 0;
            } else {
                return item->second;
            }
        }

        void convert_tokens_to_ids(const std::vector<std::string> &tokens, uint64_t *ids);

    private:
        std::unordered_map<std::string, uint64_t> *vocab;
        BasicTokenizer *basic_tokenizer;
        WordpieceTokenizer *wordpiece_tokenizer;
    };

}

#endif //CUBERT_TOKENIZATION_H
