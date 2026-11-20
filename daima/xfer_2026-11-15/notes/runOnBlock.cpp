void AffineDataCopyGeneration::runOnBlock(Block *block,
                                          DenseSet<Operation *> &copyNests) {
  if (block->empty())
    return;

  uint64_t fastMemCapacityBytes =
      fastMemoryCapacity != std::numeric_limits<uint64_t>::max()
          ? fastMemoryCapacity * 1024
          : fastMemoryCapacity;
  AffineCopyOptions copyOptions = {generateDma, slowMemorySpace,
                                   fastMemorySpace, tagMemorySpace,
                                   fastMemCapacityBytes};

  // Every affine.for op in the block starts and ends a block range for copying;
  // in addition, a contiguous sequence of operations starting with a
  // load/store op but not including any copy nests themselves is also
  // identified as a copy block range. Straightline code (a contiguous chunk of
  // operations excluding AffineForOp's) are always assumed to not exhaust
  // memory. As a result, this approach is conservative in some cases at the
  // moment; we do a check later and report an error with location info.

  // Get to the first load, store, or for op (that is not a copy nest itself).
  auto curBegin = llvm::find_if(*block, [&](Operation &op) {
    return isa<AffineLoadOp, AffineStoreOp, AffineForOp>(op) &&
           copyNests.count(&op) == 0;
  });

  // Create [begin, end) ranges.
  auto it = curBegin;
  while (it != block->end()) {
    AffineForOp forOp;
    // If you hit a non-copy for loop, we will split there.
    if ((forOp = dyn_cast<AffineForOp>(&*it)) && copyNests.count(forOp) == 0) {
      // Perform the copying up unti this 'for' op first.
      (void)affineDataCopyGenerate(/*begin=*/curBegin, /*end=*/it, copyOptions,
                                   /*filterMemRef=*/std::nullopt, copyNests);

      // Returns true if the footprint is known to exceed capacity.
      auto exceedsCapacity = [&](AffineForOp forOp) {
        std::optional<int64_t> footprint =
            getMemoryFootprintBytes(forOp,
                                    /*memorySpace=*/0);
        return (footprint.has_value() &&
                static_cast<uint64_t>(*footprint) > fastMemCapacityBytes);
      };

      // If the memory footprint of the 'affine.for' loop is higher than fast
      // memory capacity (when provided), we recurse to copy at an inner level
      // until we find a depth at which footprint fits in fast mem capacity. If
      // the footprint can't be calculated, we assume for now it fits. Recurse
      // inside if footprint for 'forOp' exceeds capacity, or when
      // skipNonUnitStrideLoops is set and the step size is not one.
      bool recurseInner = skipNonUnitStrideLoops ? forOp.getStep() != 1
                                                 : exceedsCapacity(forOp);
      if (recurseInner) {
        // We'll recurse and do the copies at an inner level for 'forInst'.
        // Recurse onto the body of this loop.
        runOnBlock(forOp.getBody(), copyNests);
      } else {
        // We have enough capacity, i.e., copies will be computed for the
        // portion of the block until 'it', and for 'it', which is 'forOp'. Note
        // that for the latter, the copies are placed just before this loop (for
        // incoming copies) and right after (for outgoing ones).

        // Inner loop copies have their own scope - we don't thus update
        // consumed capacity. The footprint check above guarantees this inner
        // loop's footprint fits.
        (void)affineDataCopyGenerate(/*begin=*/it, /*end=*/std::next(it),
                                     copyOptions,
                                     /*filterMemRef=*/std::nullopt, copyNests);
      }
      // Get to the next load or store op after 'forOp'.
      curBegin = std::find_if(std::next(it), block->end(), [&](Operation &op) {
        return isa<AffineLoadOp, AffineStoreOp, AffineForOp>(op) &&
               copyNests.count(&op) == 0;
      });
      it = curBegin;
    } else {
      assert(copyNests.count(&*it) == 0 &&
             "all copy nests generated should have been skipped above");
      // We simply include this op in the current range and continue for more.
      ++it;
    }
  }

  // Generate the copy for the final block range.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->hasTrait<OpTrait::IsTerminator>() &&
           "can't be a terminator");
    // Exclude the affine.yield - hence, the std::prev.
    (void)affineDataCopyGenerate(/*begin=*/curBegin,
                                 /*end=*/std::prev(block->end()), copyOptions,
                                 /*filterMemRef=*/std::nullopt, copyNests);
  }
}

