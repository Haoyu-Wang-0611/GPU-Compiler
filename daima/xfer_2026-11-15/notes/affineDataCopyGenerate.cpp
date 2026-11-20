mlir::affine::affineDataCopyGenerate(Block::iterator begin, Block::iterator end,
                                     const AffineCopyOptions &copyOptions,
                                     std::optional<Value> filterMemRef,
                                     DenseSet<Operation *> &copyNests) {
  if (begin == end)
    return success();

  assert(begin->getBlock() == std::prev(end)->getBlock() &&
         "Inconsistent block begin/end args");
  assert(end != end->getBlock()->end() && "end can't be the block terminator");

  Block *block = begin->getBlock();

  // Copies will be generated for this depth, i.e., symbolic in all loops
  // surrounding the this block range.
  unsigned copyDepth = getNestingDepth(&*begin);

  LDBG() << "Generating copies at depth " << copyDepth;
  LDBG() << "from begin: "
         << OpWithFlags(&*begin, OpPrintingFlags().skipRegions());
  LDBG() << "to inclusive end: "
         << OpWithFlags(&*std::prev(end), OpPrintingFlags().skipRegions());

  // List of memory regions to copy for. We need a map vector to have a
  // guaranteed iteration order to write test cases. CHECK-DAG doesn't help here
  // since the alloc's for example are identical except for the SSA id.
  SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4> readRegions;
  SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4> writeRegions;

  // Map from original memref's to the fast buffers that their accesses are
  // replaced with.
  DenseMap<Value, Value> fastBufferMap;

  // To check for errors when walking the block.
  bool error = false;

  // Walk this range of operations  to gather all memory regions.
  block->walk(begin, end, [&](Operation *opInst) {
    Value memref;
    MemRefType memrefType;
    // Gather regions to allocate to buffers in faster memory space.
    if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
      memref = loadOp.getMemRef();
      memrefType = loadOp.getMemRefType();
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
      memref = storeOp.getMemRef();
      memrefType = storeOp.getMemRefType();
    }
    // Not an affine.load/store op.
    if (!memref)
      return;

    if ((filterMemRef.has_value() && filterMemRef != memref) ||
        (isa_and_nonnull<IntegerAttr>(memrefType.getMemorySpace()) &&
         memrefType.getMemorySpaceAsInt() != copyOptions.slowMemorySpace))
      return;

    if (!memref.getParentRegion()->isAncestor(block->getParent())) {
      LDBG() << "memref definition is inside of the depth at "
             << "which copy-in/copy-out would happen";
      return;
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(region->compute(opInst, copyDepth, /*sliceState=*/nullptr,
                               /*addMemRefDimBounds=*/false))) {
      LDBG() << "Error obtaining memory region: semi-affine maps?";
      LDBG() << "over-approximating to the entire memref";
      if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
        LDBG() << "non-constant memref sizes not yet supported";
        error = true;
        return;
      }
    }

    // Each memref has a single buffer associated with it irrespective of how
    // many load's and store's happen on it.
    // TODO: in the future, when regions don't intersect and satisfy
    // other properties (based on load/store regions), we could consider
    // multiple buffers per memref.

    // Add to the appropriate region if it's not already in it, or take a
    // bounding box union with the existing one if it's already in there.
    // Note that a memref may have both read and write regions - so update the
    // region in the other list if one exists (write in case of read and vice
    // versa) since there is a single bounding box for a memref across all reads
    // and writes that happen on it.

    // Attempts to update; returns true if 'region' exists in targetRegions.
    auto updateRegion =
        [&](const SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4>
                &targetRegions) {
          const auto *const it = targetRegions.find(region->memref);
          if (it == targetRegions.end())
            return false;

          // Perform a union with the existing region.
          if (failed(it->second->unionBoundingBox(*region))) {
            LDBG() << "Memory region bounding box failed; "
                   << "over-approximating to the entire memref";
            // If the union fails, we will overapproximate.
            if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
              LDBG() << "non-constant memref sizes not yet supported";
              error = true;
              return true;
            }
            it->second->getConstraints()->clearAndCopyFrom(
                *region->getConstraints());
          } else {
            // Union was computed and stored in 'it->second': copy to 'region'.
            region->getConstraints()->clearAndCopyFrom(
                *it->second->getConstraints());
          }
          return true;
        };

    bool existsInRead = updateRegion(readRegions);
    if (error)
      return;
    bool existsInWrite = updateRegion(writeRegions);
    if (error)
      return;

    // Finally add it to the region list.
    if (region->isWrite() && !existsInWrite) {
      writeRegions[region->memref] = std::move(region);
    } else if (!region->isWrite() && !existsInRead) {
      readRegions[region->memref] = std::move(region);
    }
  });

  if (error) {
    LDBG() << "copy generation failed for one or more memref's in this block";
    return failure();
  }

  uint64_t totalCopyBuffersSizeInBytes = 0;
  bool ret = true;
  auto processRegions =
      [&](const SmallMapVector<Value, std::unique_ptr<MemRefRegion>, 4>
              &regions) {
        for (const auto &regionEntry : regions) {
          // For each region, hoist copy in/out past all hoistable
          // 'affine.for's.
          Block::iterator copyInPlacementStart, copyOutPlacementStart;
          Block *copyPlacementBlock;
          findHighestBlockForPlacement(
              *regionEntry.second, *block, begin, end, &copyPlacementBlock,
              &copyInPlacementStart, &copyOutPlacementStart);

          uint64_t sizeInBytes;
          Block::iterator nBegin, nEnd;
          LogicalResult iRet = generateCopy(
              *regionEntry.second, block, begin, end, copyPlacementBlock,
              copyInPlacementStart, copyOutPlacementStart, copyOptions,
              fastBufferMap, copyNests, &sizeInBytes, &nBegin, &nEnd);
          if (succeeded(iRet)) {
            // begin/end could have been invalidated, and need update.
            begin = nBegin;
            end = nEnd;
            totalCopyBuffersSizeInBytes += sizeInBytes;
          }
          ret = ret & succeeded(iRet);
        }
      };
  processRegions(readRegions);
  processRegions(writeRegions);

  if (!ret) {
    LDBG() << "copy generation failed for one or more memref's in this block";
    return failure();
  }

  // For a range of operations, a note will be emitted at the caller.
  AffineForOp forOp;
  if (llvm::DebugFlag && (forOp = dyn_cast<AffineForOp>(&*begin))) {
    LLVM_DEBUG(forOp.emitRemark()
               << llvm::divideCeil(totalCopyBuffersSizeInBytes, 1024)
               << " KiB of copy buffers in fast memory space for this block");
  }

  if (totalCopyBuffersSizeInBytes > copyOptions.fastMemCapacityBytes) {
    block->getParentOp()->emitWarning(
        "total size of all copy buffers' for this block exceeds fast memory "
        "capacity");
  }

  return success();
}

// A convenience version of affineDataCopyGenerate for all ops in the body of
// an AffineForOp.
