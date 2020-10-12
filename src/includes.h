#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

#include "R.h"
#include "find_nan.h"
#include "mh_adapt.h"
#include "field_v_concatm.h"
#include "caching_pairwise_compare.h"
#include "covariance_functions.h"
#include "debug.h"

#include "mgp_utils.h"

