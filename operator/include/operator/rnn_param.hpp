/*
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
 * Copyright (c) 2019, Open AI Lab
 * Author: zpluo@openailab.com
 */
#ifndef __RNN_PARAM_HPP__
#define __RNN_PARAM_HPP__

#include <vector>

#include "parameter.hpp"

namespace TEngine {

#define RNN_ACT_TANH 1

struct RNNParam : public NamedParam
{
    float clip;
    int output_len;
    int sequence_len;
    int input_size;
    int hidden_size;
    int has_clip;
    int has_bias;
    int has_init_state;
    int activation;

    DECLARE_PARSER_STRUCTURE(RNNParam)
    {
        DECLARE_PARSER_ENTRY(clip);
        DECLARE_PARSER_ENTRY(output_len);
        DECLARE_PARSER_ENTRY(sequence_len);
        DECLARE_PARSER_ENTRY(input_size);
        DECLARE_PARSER_ENTRY(hidden_size);
        DECLARE_PARSER_ENTRY(has_clip);
        DECLARE_PARSER_ENTRY(has_bias);
        DECLARE_PARSER_ENTRY(has_init_state);
        DECLARE_PARSER_ENTRY(activation);
    };
};

}    // namespace TEngine

#endif
