/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/dem.h"
#include "common/DeviceCodeRegistry.h"
#include "common/ExecutionContext.h"
#include "nvqir/dem/DemScope.h"
#include <stdexcept>

namespace cudaq::detail {

cudaq::dem_result launchDemPolicy(const cudaq::dem_policy &policy,
                                  cudaq::ExecutionContext &ctx,
                                  const dem_policy_launcher &launchPolicy,
                                  const std::string &plugin_name) {
  // NOTE: measurement-conditioned Paulis (`if (m) P(q)`) are lowered to
  // symbolic feedback (`qec.apply_pauli_feedback`) on the DEM path (see
  // Compiler::executeMainPipeline / the `lower-pauli-feedback` pass), so they
  // are representable in the detector error model. Any conditional the pass
  // cannot lower stays a concrete branch, matching prior behavior.

  // RAII: claim the thread-local analysis-simulator slot backed by the
  // @p plugin_name plugin. The scope starts from a clean simulator and
  // releases the override on every exit path.
  auto demScope = nvqir::dem::make_scope(plugin_name);
  return launchPolicy(policy, ctx);
}

} // namespace cudaq::detail
