/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cmath>    // for pow

#include "ErrorMacros.h"

using std::endl;

#define DLA_MAX_TRUNCATE_SIZE   (std::pow(2,6) - 1)

// support api's for int8 quantization and scaling/rescaling

template <typename SC, typename SH>
NvDlaError scaleAndShiftFromScalarImpl1
(
    NvF32 scalar,
    std::pair<SC, SH>* scaleAndShift,
    NvU32 powerOfTwo = 0
)
{
    NvDlaError e = NvDlaSuccess;

    const bool debug = false;

    NvS32 scale = 1;
    NvU32 attempts = 0;

    bool found = false;
    bool isNeg = scalar < 0;
    bool sclInRange = true;
    bool shftInRange = true;

    NvF32 approxValue = 0.0f;
    NvF32 absScalar = fabs(scalar);
    NvF32 closestFound = 0.0f;
    NvF32 toleranceMax = std::pow(2, 4);

    if (absScalar > 1)
    {
        bool isNoFractional = (absScalar == floor(absScalar));
        if (isNoFractional && (absScalar <= std::numeric_limits<SC>::max()))
        {
            scale = SC(scalar);
            powerOfTwo = 0;
            found = true;
        }
        else
        {
            scale       = std::numeric_limits<SC>::max();
            // use suggested powerOf2 (if any) and adjust scale to meet the scalar
            powerOfTwo  = powerOfTwo != 0 ? powerOfTwo : ceil(log(NvF32(absScalar / scale))/log(2));

            NvF32 tolerance = 1;
            do {
                do
                {
                    approxValue = NvF32(scale) * std::pow(2, powerOfTwo);
                    closestFound = std::min<NvF32>(fabs(closestFound - absScalar), fabs(approxValue - absScalar)) ==
                                   fabs(closestFound - absScalar) ?
                                   closestFound : approxValue;
                    if (fabs(approxValue - absScalar) <= tolerance)
                    {
                        found = true;
                        break;
                    }
                    else
                    {
                        scale--;
                    }

                    sclInRange  = (scale <= std::numeric_limits<SC>::max()) &&
                                  (scale >= std::numeric_limits<SC>::min());
                    shftInRange = (powerOfTwo <= std::numeric_limits<SH>::max()) &&
                                  (powerOfTwo >= std::numeric_limits<SH>::min());
                    attempts++;
                } while (attempts != 1000 && sclInRange && shftInRange);

                if (found)
                    break;

                // reset stats and retry
                tolerance  += 1;
                attempts    = 0;
                scale       = std::numeric_limits<SC>::max();
                powerOfTwo  = powerOfTwo != 0 ? powerOfTwo : ceil(log(NvF32(absScalar / scale))/log(2));
            } while (tolerance <= toleranceMax);
        }

        if (found)
        {
            if ( debug )
            {
                gLogInfo << scalar << " = " << (isNeg ? -scale : scale) << "*2^"
                         << powerOfTwo << " [" << approxValue << "]" << std::endl;
            }
            scale *= isNeg ? -1 : 1;
            *scaleAndShift = std::make_pair<SC, SH>(SC(scale), SH(powerOfTwo));
        }
        else
        {
            gLogWarning << "Couldn't converge on `2^(ls) * m` which could safely represent " << scalar
                        << " within acceptable tolerance of +/-" << toleranceMax
                        << " using closest found: " << closestFound << "instead" << endl;
            scale  = closestFound / std::pow(2, powerOfTwo);
            scale *= isNeg ? -1 : 1;
            *scaleAndShift = std::make_pair<SC, SH>(SC(scale), SH(powerOfTwo));
        }
    }
    else
    {
        scale = 1;
        powerOfTwo = 0;
        NvF32 tolerance = 10e-8;
        NvF32 toleranceMax = 0.1;
        do {
            do {
                approxValue = NvF32(scale) / std::pow(2, powerOfTwo);
                closestFound = std::min<NvF32>(fabs(closestFound - absScalar), fabs(approxValue - absScalar)) ==
                               fabs(closestFound - absScalar) ? closestFound : approxValue;
                if (fabs(approxValue - absScalar) <= tolerance)
                {
                    found = true;
                    break;
                }

                if (approxValue > absScalar)
                {
                    powerOfTwo++;
                }
                else if (approxValue < absScalar)
                {
                    scale++;
                }

                sclInRange  = (scale <= std::numeric_limits<SC>::max()) &&
                              (scale >= std::numeric_limits<SC>::min());
                shftInRange = (powerOfTwo <= std::numeric_limits<SH>::max()) &&
                              (powerOfTwo >= std::numeric_limits<SH>::min());
                attempts++;
            } while (attempts != 1000 && sclInRange && shftInRange);

            if (found)
                break;

            // reset stats and retry
            tolerance  *= 10;
            attempts    = 0;
            scale       = 1;
            powerOfTwo  = 0;
        } while (tolerance <= toleranceMax);

        if (found)
        {
            if ( debug )
            {
                gLogInfo << scalar << " = " << (isNeg ? -scale : scale) << "/(2^"
                         << powerOfTwo << ") [" << approxValue << "]" << std::endl;
            }
            scale *= -isNeg ? -1 : 1;
            *scaleAndShift = std::make_pair<SC, SH>(SC((isNeg ? -scale : scale)), SH(powerOfTwo));
        }
        else
        {
            gLogWarning << "Couldn't converge on `2^(-t) * s` which could safely represent " << scalar
                        << " within acceptable tolerance of +/-" << toleranceMax
                        << " using closest found: " << closestFound << "instead" << endl;
            scale  = closestFound * std::pow(2, powerOfTwo);
            scale *= isNeg ? -1 : 1;
            *scaleAndShift = std::make_pair<SC, SH>(SC(scale), SH(powerOfTwo));
        }
    }

    if (!found)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Couldn't converge on `2^(x) * y` which could "
                                "safely represent %f within acceptable tolerance\n", scalar);
    }

fail:
    return e;
}

/**
 * All scalar values (S_fp) can be represented as n * (2 ^ -m) if within range
 *
 * S_fp = n_i16 * (2 ^ -m), such that
 *          m = 15 - b',
 *      n_i16 = INT16(S_fp * (2 ^ (15 - b')))
 * where b' represent #bits to tranverse from MSB before radix point (in binary)
 *  of S_fp is hit.
 *
 * Since we can only have signed 16 bit scale/numerator, limiting the right shift
 * to 15 bits. With a larger divisor, we lose precision and small scales become 0
 * With a smaller divisor, then the largest value overflows. If the numerator/scale
 * overflows NvS16 bynamic range, compilation should fail.
 *
 * todo: for dla2, add constraints on max shift/truncate width
 **/
template <typename SC, typename SH>
NvDlaError scaleAndShiftFromScalarImpl2
(
    NvF32 scalar,
    std::pair<SC, SH>* scaleAndShift,
    //NvU32 maxShiftWidth,
    NvU32 powerOfTwo = 0
)
{
    NvDlaError e = NvDlaSuccess;

    const NvS32 MIN_TOLERABLE_SCALE = std::pow(2,1);

    NvF32 absScalar = fabs(scalar);
    bool isNeg  = scalar < 0;
    NvS32 scale = 0;
    NvS32 numBits = 0;

    // Handle special case of scalar being zero.
    if(absScalar == 0)
    {
        powerOfTwo = powerOfTwo != 0 ? powerOfTwo : 1; // Any value would do if none provided
        scale = 0;
    }
    else
    {
        // Find the number of bits required for non-fractional part
        numBits = floor(log(absScalar)/log(2)) + 1;

        // Check if it is within range
        if (powerOfTwo == 0 && numBits > 15)
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Scale value for %f is "
                    "beyond dynamic range of NvS16\n", scalar);
        }

        // Update truncateFactor (powerOfTwo) and scale based on numBits
        powerOfTwo = powerOfTwo != 0 ? powerOfTwo : 15 - numBits;

        if (powerOfTwo > 31 && powerOfTwo <= 63)
        {
            /* Warn user on high truncate value */
            REPORT_ERROR(NvDlaError_BadParameter, "Truncate too high: %d\n", powerOfTwo);
        }
        else if (powerOfTwo >= 64)
        {
            /* Error out as impossible to program */
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue,
                                "Truncate value %d is out of range [0,63]",
                                powerOfTwo);
        }

        scale = std::pow(2, powerOfTwo) * absScalar;
        if (scale == 0)
        {
            // TODO: PROPAGATE_ERROR_FAIL once everything in place
            REPORT_ERROR(NvDlaError_BadValue, "Scale value is 0 for %f\n", scalar);
        }
        else if (scale < MIN_TOLERABLE_SCALE)
        {
            gLogWarning << "Scale value " << scale << " for " << scalar << " is too small "
                << "(threshold: " << MIN_TOLERABLE_SCALE << ")" << endl;
        }
        else if (scale > std::numeric_limits<SC>::max())
        {
            PROPAGATE_ERROR_FAIL(NvDlaError_BadValue, "Scale value %d for %f is "
                    "beyond dynamic range of NvS16\n", scale, scalar);
        }
    }

    *scaleAndShift = std::make_pair<SC, SH>(SC((isNeg ? -scale : scale)), SH(powerOfTwo));

fail:
    return e;
}


template <typename SC, typename SH>
NvDlaError calculateScaleAndShiftFromScalar
(
    NvF32 scalar,
    std::pair<SC, SH>* scaleAndShift,
    NvU32 powerOfTwo = 0
)
{
    NvDlaError e = NvDlaSuccess;

    // toggle between the 2 implementations in case of debugging Impl2 is safe for now (mnist)
    // PROPAGATE_ERROR_FAIL( scaleAndShiftFromScalarImpl1(scalar, scaleAndShift, powerOfTwo) );
    PROPAGATE_ERROR_FAIL( scaleAndShiftFromScalarImpl2(scalar, scaleAndShift, powerOfTwo) );

fail:
    return e;
}

template <typename SC, typename SH>
NvDlaError factorizeScalars
(
    std::vector<NvF32> scalars,
    std::vector< std::pair<SC, SH> >* scalesAndShifts,
    NvU32 commonPowerOfTwo = 0
)
{
    NvDlaError e = NvDlaSuccess;

    for (NvU32 ss = 0; ss < scalars.size(); ++ss)
    {
        std::pair<SC, SH> sclAndShft;
        e = calculateScaleAndShiftFromScalar<SC, SH>(scalars[ss], &sclAndShft, commonPowerOfTwo);
        if (e != NvDlaSuccess)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, " Couldn't converge on `2^(x) * y` which could "
                                    "safely represent %f within acceptable tolerance\n", scalars[ss]);
        }
        scalesAndShifts->push_back(sclAndShft);
    }

fail:
    return e;
}
