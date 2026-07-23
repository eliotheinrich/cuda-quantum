/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERPAULIFEEDBACK
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Peel a chain of local `cc.load`s back to the value most recently `cc.store`d
/// into the pointer (the temporary the C++ frontend emits for a `measure_result`
/// / `bool` variable). Returns @p v unchanged if there is nothing to peel.
static Value peelStoreLoad(Value v) {
  while (auto load = v.getDefiningOp<cudaq::cc::LoadOp>()) {
    Value ptr = load.getPtrvalue();
    Value stored;
    for (Operation *user : ptr.getUsers())
      if (auto st = dyn_cast<cudaq::cc::StoreOp>(user))
        stored = st.getValue();
    if (!stored)
      return v;
    v = stored;
  }
  return v;
}

/// Decompose the `i1` condition of a `cc.if` into an affine GF(2) form:
/// a constant parity XOR a set of measurement-result bits. `if (m)` -> {0,[m]};
/// `if (not m)` -> {1,[m]}; `if (m0 xor m1)` -> {0,[m0,m1]}. This is exactly the
/// class representable as Stim feedback, since feedback is a linear frame
/// update (`q ^= rec[k]`) that composes by XOR and an affine constant maps to an
/// unconditional Pauli. Non-linear conditions (`and`/`or`, which carry a `m0*m1`
/// product term) and any other op make this return false.
static bool traceAffineCondition(Value cond, bool &parity,
                                 SmallVector<Value> &handles) {
  cond = peelStoreLoad(cond);
  if (auto disc = cond.getDefiningOp<cudaq::quake::DiscriminateOp>()) {
    Value h = peelStoreLoad(disc.getMeasurement());
    if (!isa<cudaq::cc::MeasureHandleType>(h.getType()))
      return false;
    handles.push_back(h);
    return true;
  }
  if (auto x = cond.getDefiningOp<mlir::arith::XOrIOp>())
    return traceAffineCondition(x.getLhs(), parity, handles) &&
           traceAffineCondition(x.getRhs(), parity, handles);
  // On `i1` operands, `!=` is XOR and `==` is XNOR (`1 xor a xor b`); both
  // affine. (`m0 != m1` lowers to `arith.cmpi ne`, not `arith.xori`.)
  if (auto cmp = cond.getDefiningOp<mlir::arith::CmpIOp>()) {
    if (!cmp.getLhs().getType().isInteger(1))
      return false;
    auto pred = cmp.getPredicate();
    if (pred == mlir::arith::CmpIPredicate::eq)
      parity = !parity;
    else if (pred != mlir::arith::CmpIPredicate::ne)
      return false;
    return traceAffineCondition(cmp.getLhs(), parity, handles) &&
           traceAffineCondition(cmp.getRhs(), parity, handles);
  }
  if (auto c = cond.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto ia = dyn_cast<mlir::IntegerAttr>(c.getValue());
    if (!ia)
      return false;
    if (!ia.getValue().isZero()) // constant `true` flips the parity
      parity = !parity;
    return true;
  }
  return false;
}

/// 0 = X, 1 = Y, 2 = Z if `op` is a single Pauli gate, else nullopt.
static std::optional<int32_t> pauliCode(Operation *op) {
  if (isa<cudaq::quake::XOp>(op))
    return 0;
  if (isa<cudaq::quake::YOp>(op))
    return 1;
  if (isa<cudaq::quake::ZOp>(op))
    return 2;
  return std::nullopt;
}

struct LowerPauliFeedbackPass
    : public cudaq::opt::impl::LowerPauliFeedbackBase<LowerPauliFeedbackPass> {
  using LowerPauliFeedbackBase::LowerPauliFeedbackBase;

  void runOnOperation() override {
    auto func = getOperation();
    SmallVector<cudaq::cc::IfOp> candidates;
    func.walk([&](cudaq::cc::IfOp ifOp) { candidates.push_back(ifOp); });

    for (auto ifOp : candidates) {
      // Shape: `cc.if(<affine cond>) { <one or more single-qubit Paulis> }`
      // with no (non-empty) else, where the condition is an XOR/negation of
      // measurement results. The AST bridge emits a one-armed `if` as an `else`
      // region holding an empty block (terminator only), so `hasElse()` -- just
      // `!elseRegion.empty()` -- is true even though nothing happens on the else
      // path. Reject only an else that does real work.
      if (ifOp.hasElse()) {
        mlir::Region &elseRegion = ifOp.getElseRegion();
        if (!elseRegion.hasOneBlock() ||
            !elseRegion.front().without_terminator().empty())
          continue;
      }
      if (!ifOp.hasThen() || !ifOp.getThenRegion().hasOneBlock())
        continue;
      Block &blk = ifOp.getThenRegion().front();

      // The then-region must hold one or more single-qubit Pauli gates (each
      // becomes its own feedback op, preserving program order) and, otherwise,
      // only pure value computations -- typically `quake.extract_ref`s for the
      // target qubits, hoisted into the branch when a qubit is first used here.
      // Those get hoisted back out below so the feedback ops can reference the
      // targets. Any impure non-Pauli op (a non-Pauli gate, a controlled gate,
      // a nested feedback op, ...) disqualifies the whole `if`, which is then
      // rejected downstream rather than silently mis-lowered.
      SmallVector<cudaq::quake::OperatorInterface> gates;
      bool ok = true;
      for (Operation &o : blk.without_terminator()) {
        auto g = dyn_cast<cudaq::quake::OperatorInterface>(&o);
        if (!g) {
          if (!isMemoryEffectFree(&o)) {
            ok = false;
            break;
          }
          continue; // pure non-gate op (e.g. extract_ref)
        }
        if (!pauliCode(g.getOperation()) || g.getControls().size() != 0 ||
            g.getTargets().size() != 1 ||
            !isa<cudaq::quake::RefType>(g.getTargets()[0].getType())) {
          ok = false;
          break;
        }
        gates.push_back(g);
      }
      if (!ok || gates.empty())
        continue;

      // The condition must be an affine (XOR/negation) function of measurement
      // results -- the only class Stim feedback can represent. Non-linear
      // conditions (`and`/`or`) fail here, leaving the `if` for the DEM-path
      // guard to reject.
      bool parity = false;
      SmallVector<Value> handles;
      if (!traceAffineCondition(ifOp.getCondition(), parity, handles))
        continue;

      // Hoist each target's pure defining chain (e.g. `quake.extract_ref`) out
      // of the branch so the new ops, inserted before the `if`, can reference
      // the targets. Bail if any producer can't be safely hoisted.
      std::function<bool(Operation *)> hoist = [&](Operation *def) -> bool {
        if (!def || def->getBlock() != &blk)
          return true; // already dominates the `if`
        if (!isMemoryEffectFree(def))
          return false;
        for (Value operand : def->getOperands())
          if (!hoist(operand.getDefiningOp()))
            return false;
        def->moveBefore(ifOp);
        return true;
      };
      for (auto g : gates)
        if (!hoist(g.getTargets()[0].getDefiningOp())) {
          ok = false;
          break;
        }
      if (!ok)
        continue;

      // Rewrite `if (c ^ m0 ^ m1 ^ ...) { P q; ... }` for each Pauli `P q`:
      //   q ^= (c ^ m0 ^ m1 ^ ...)
      // = an unconditional `P q` when the constant parity `c` is 1, plus one
      //   `apply_pauli_feedback` (i.e. `CP rec[-k] q`) per measurement handle.
      // These all act with the same single-qubit Pauli, so they commute and
      // their order is irrelevant; distinct gates keep program order.
      OpBuilder builder(ifOp);
      for (auto g : gates) {
        Value target = g.getTargets()[0];
        auto pauliAttr = builder.getI32IntegerAttr(*pauliCode(g.getOperation()));
        if (parity)
          builder.clone(*g.getOperation()); // unconditional Pauli (the `c` term)
        for (Value h : handles)
          cudaq::qec::ApplyPauliFeedbackOp::create(builder, g.getLoc(), h,
                                                   target, pauliAttr);
      }
      ifOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createLowerPauliFeedback() {
  return std::make_unique<LowerPauliFeedbackPass>();
}
