static LogicalResult generateCopy(
    const MemRefRegion &region, Block *block, Block::iterator begin,
    Block::iterator end, Block *copyPlacementBlock,
    Block::iterator copyInPlacementStart, Block::iterator copyOutPlacementStart,
    const AffineCopyOptions &copyOptions, DenseMap<Value, Value> &fastBufferMap,
    DenseSet<Operation *> &copyNests, uint64_t *sizeInBytes,
    Block::iterator *nBegin, Block::iterator *nEnd) {
  *nBegin = begin;
  *nEnd = end;

  auto f = begin->getParentOfType<FunctionOpInterface>();
  OpBuilder topBuilder(f.getFunctionBody());
  Value zeroIndex = arith::ConstantIndexOp::create(topBuilder, f.getLoc(), 0);

  *sizeInBytes = 0;

  if (begin == end)
    return success();

  // Record the last op in the block for which we are performing copy
  // generation. We later do the memref replacement only in [begin, lastCopyOp]
  // so that the original memref's used in the data movement code themselves
  // don't get replaced.
  Operation *lastCopyOp = end->getPrevNode();

  // Is the copy out point at the end of the block where we are doing
  // explicit copying.
  bool isCopyOutAtEndOfBlock = (end == copyOutPlacementStart);

  // Copies for read regions are going to be inserted at 'begin'.
  OpBuilder prologue(copyPlacementBlock, copyInPlacementStart);
  // Copies for write regions are going to be inserted at 'end'.
  OpBuilder epilogue(copyPlacementBlock, copyOutPlacementStart);
  OpBuilder &b = region.isWrite() ? epilogue : prologue;

  // Builder to create constants at the top level.
  auto func =
      copyPlacementBlock->getParent()->getParentOfType<FunctionOpInterface>();
  OpBuilder top(func.getFunctionBody());

  auto loc = region.loc;
  auto memref = region.memref;
  auto memRefType = cast<MemRefType>(memref.getType());

  if (!memRefType.getLayout().isIdentity()) {
    LDBG() << "Non-identity layout map not yet supported";
    return failure();
  }

  // Indices to use for the copying.
  // Indices for the original memref being copied from/to.
  SmallVector<Value, 4> memIndices;
  // Indices for the faster buffer being copied into/from.
  SmallVector<Value, 4> bufIndices;

  unsigned rank = memRefType.getRank();
  if (rank == 0) {
    LDBG() << "Non-zero ranked memrefs supported";
    return failure();
  }

  SmallVector<int64_t, 4> fastBufferShape;

  // Compute the extents of the buffer.
  SmallVector<AffineMap, 2> lbs;
  lbs.reserve(rank);
  std::optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&fastBufferShape, &lbs);
  if (!numElements) {
    LDBG() << "Non-constant region size not supported";
    return failure();
  }

  if (llvm::any_of(lbs, [](AffineMap lb) { return lb.getNumResults() > 1; })) {
    // This can be supported in the future if needed.
    LDBG() << "Max lower bound for memref region start not supported";
    return failure();
  }

  if (*numElements == 0) {
    LDBG() << "Nothing to copy";
    return success();
  }

  SmallVector<AffineMap, 4> lbMaps(rank), ubMaps(rank);
  for (unsigned i = 0; i < rank; ++i) {
    region.getLowerAndUpperBound(i, lbMaps[i], ubMaps[i]);
    if (lbMaps[i].getNumResults() == 0 || ubMaps[i].getNumResults() == 0) {
      LDBG() << "Missing lower or upper bound for region along dimension: "
             << i;
      return failure();
    }
  }

  const FlatAffineValueConstraints *cst = region.getConstraints();
  // 'regionSymbols' hold values that this memory region is symbolic/parametric
  // on; these typically include loop IVs surrounding the level at which the
  // copy generation is being done or other valid symbols in MLIR.
  SmallVector<Value, 8> regionSymbols;
  cst->getValues(rank, cst->getNumVars(), &regionSymbols);

  // Construct the access expression for the fast memory buffer. The access
  // expression for a particular dimension of the fast buffer is obtained by
  // subtracting out the lower bound on the original memref's data region
  // along the corresponding dimension.

  // Index start offsets for faster memory buffer relative to the original.
  SmallVector<AffineExpr, 4> fastBufOffsets;
  fastBufOffsets.reserve(rank);
  for (unsigned d = 0; d < rank; d++) {
    assert(lbs[d].getNumSymbols() == cst->getNumCols() - rank - 1 &&
           "incorrect bound size");

    // Set copy start location for this dimension in the lower memory space
    // memref.
    if (lbs[d].isSingleConstant()) {
      auto indexVal = lbs[d].getSingleConstantResult();
      if (indexVal == 0) {
        memIndices.push_back(zeroIndex);
      } else {
        memIndices.push_back(
            arith::ConstantIndexOp::create(top, loc, indexVal).getResult());
      }
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      // Remap all inputs of the map to dimensions uniformly since in the
      // generate IR we need valid affine symbols as opposed to "symbols" for
      // the purpose of the memref region.
      SmallVector<AffineExpr> symReplacements(lbs[d].getNumSymbols());
      for (unsigned i = 0, e = lbs[d].getNumSymbols(); i < e; ++i)
        symReplacements[i] = top.getAffineDimExpr(i);
      lbs[d] = lbs[d].replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symReplacements, lbs[d].getNumSymbols(),
          /*numResultSyms=*/0);
      memIndices.push_back(
          AffineApplyOp::create(b, loc, lbs[d], regionSymbols));
    }
    // The fast buffer is copied into at location zero; addressing is relative.
    bufIndices.push_back(zeroIndex);

    // Record the offsets since they are needed to remap the memory accesses of
    // the original memref further below.
    fastBufOffsets.push_back(lbs[d].getResult(0));
  }

  // The faster memory space buffer.
  Value fastMemRef;

  // Check if a buffer was already created.
  bool existingBuf = fastBufferMap.count(memref) > 0;
  if (!existingBuf) {
    AffineMap fastBufferLayout = b.getMultiDimIdentityMap(rank);
    auto fastMemRefType =
        MemRefType::get(fastBufferShape, memRefType.getElementType(),
                        fastBufferLayout, copyOptions.fastMemorySpace);

    // Create the fast memory space buffer just before the 'affine.for'
    // operation.
    fastMemRef =
        memref::AllocOp::create(prologue, loc, fastMemRefType).getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    auto maySizeInBytes = getIntOrFloatMemRefSizeInBytes(fastMemRefType);
    // We don't account for things of unknown size.
    *sizeInBytes = maySizeInBytes.value_or(0);

    LLVM_DEBUG(emitRemarkForBlock(*block)
               << "Creating fast buffer of type " << fastMemRefType
               << " and size " << llvm::divideCeil(*sizeInBytes, 1024)
               << " KiB\n");
  } else {
    // Reuse the one already created.
    fastMemRef = fastBufferMap[memref];
  }

  auto numElementsSSA = arith::ConstantIndexOp::create(top, loc, *numElements);

  Value dmaStride;
  Value numEltPerDmaStride;
  if (copyOptions.generateDma) {
    SmallVector<StrideInfo, 4> dmaStrideInfos;
    getMultiLevelStrides(region, fastBufferShape, &dmaStrideInfos);

    // TODO: use all stride levels once DmaStartOp is extended for
    // multi-level strides.
    if (dmaStrideInfos.size() > 1) {
      LDBG() << "Only up to one level of stride supported";
      return failure();
    }

    if (!dmaStrideInfos.empty()) {
      dmaStride =
          arith::ConstantIndexOp::create(top, loc, dmaStrideInfos[0].stride);
      numEltPerDmaStride = arith::ConstantIndexOp::create(
          top, loc, dmaStrideInfos[0].numEltPerStride);
    }
  }

  // Create fully composed affine maps for each memref.
  auto memAffineMap = b.getMultiDimIdentityMap(memIndices.size());
  fullyComposeAffineMapAndOperands(&memAffineMap, &memIndices);
  auto bufAffineMap = b.getMultiDimIdentityMap(bufIndices.size());
  fullyComposeAffineMapAndOperands(&bufAffineMap, &bufIndices);

  if (!copyOptions.generateDma) {
    // Point-wise copy generation.
    auto copyNest =
        generatePointWiseCopy(loc, memref, fastMemRef, lbMaps,
                              /*lbOperands=*/regionSymbols, ubMaps,
                              /*ubOperands=*/regionSymbols, fastBufOffsets,
                              /*isCopyOut=*/region.isWrite(), b);

    // Record this so that we can skip it from yet another copy.
    copyNests.insert(copyNest);

    // Since new ops are being appended (for copy out's), adjust the end to
    // mark end of block range being processed if necessary.
    if (region.isWrite() && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(copyNest.getOperation());
  } else {
    // DMA generation.
    // Create a tag (single element 1-d memref) for the DMA.
    auto tagMemRefType = MemRefType::get({1}, top.getIntegerType(32), {},
                                         copyOptions.tagMemorySpace);
    auto tagMemRef = memref::AllocOp::create(prologue, loc, tagMemRefType);

    SmallVector<Value, 4> tagIndices({zeroIndex});
    auto tagAffineMap = b.getMultiDimIdentityMap(tagIndices.size());
    fullyComposeAffineMapAndOperands(&tagAffineMap, &tagIndices);
    if (!region.isWrite()) {
      // DMA non-blocking read from original buffer to fast buffer.
      AffineDmaStartOp::create(b, loc, memref, memAffineMap, memIndices,
                               fastMemRef, bufAffineMap, bufIndices, tagMemRef,
                               tagAffineMap, tagIndices, numElementsSSA,
                               dmaStride, numEltPerDmaStride);
    } else {
      // DMA non-blocking write from fast buffer to the original memref.
      auto op = AffineDmaStartOp::create(
          b, loc, fastMemRef, bufAffineMap, bufIndices, memref, memAffineMap,
          memIndices, tagMemRef, tagAffineMap, tagIndices, numElementsSSA,
          dmaStride, numEltPerDmaStride);
      // Since new ops may be appended at 'end' (for outgoing DMAs), adjust the
      // end to mark end of block range being processed.
      if (isCopyOutAtEndOfBlock)
        *nEnd = Block::iterator(op.getOperation());
    }

    // Matching DMA wait to block on completion; tag always has a 0 index.
    AffineDmaWaitOp::create(b, loc, tagMemRef, tagAffineMap, zeroIndex,
                            numElementsSSA);

    // Generate dealloc for the tag.
    auto tagDeallocOp = memref::DeallocOp::create(epilogue, loc, tagMemRef);
    if (*nEnd == end && isCopyOutAtEndOfBlock)
      // Since new ops are being appended (for outgoing DMAs), adjust the end to
      // mark end of range of the original.
      *nEnd = Block::iterator(tagDeallocOp.getOperation());
  }

  // Generate dealloc for the buffer.
  if (!existingBuf) {
    auto bufDeallocOp = memref::DeallocOp::create(epilogue, loc, fastMemRef);
    // When generating pointwise copies, `nEnd' has to be set to deallocOp on
    // the fast buffer (since it marks the new end insertion point).
    if (!copyOptions.generateDma && *nEnd == end && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(bufDeallocOp.getOperation());
  }

  // Replace all uses of the old memref with the faster one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  // Ex: to replace load %A[%i, %j] with load %Abuf[%i - %iT, %j - %jT],
  // index remap will be (%i, %j) -> (%i - %iT, %j - %jT),
  // i.e., affine.apply (d0, d1, d2, d3) -> (d2-d0, d3-d1) (%iT, %jT, %i, %j),
  // and (%iT, %jT) will be the 'extraOperands' for 'rep all memref uses with'.
  // d2, d3 correspond to the original indices (%i, %j).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    // The starting operands of indexRemap will be regionSymbols (the symbols on
    // which the memref region is parametric); then those corresponding to
    // the memref's original indices follow.
    auto dimExpr = b.getAffineDimExpr(regionSymbols.size() + i);
    remapExprs.push_back(dimExpr - fastBufOffsets[i]);
  }
  auto indexRemap = AffineMap::get(regionSymbols.size() + rank, 0, remapExprs,
                                   b.getContext());

  // Record the begin since it may be invalidated by memref replacement.
  Block::iterator prevOfBegin;
  bool isBeginAtStartOfBlock = (begin == block->begin());
  if (!isBeginAtStartOfBlock)
    prevOfBegin = std::prev(begin);

  auto userFilterFn = [&](Operation *user) {
    auto *ancestorUser = block->findAncestorOpInBlock(*user);
    return ancestorUser && !ancestorUser->isBeforeInBlock(&*begin) &&
           !lastCopyOp->isBeforeInBlock(ancestorUser);
  };

  // *Only* those uses within the range [begin, end) of 'block' are replaced.
  (void)replaceAllMemRefUsesWith(memref, fastMemRef,
                                 /*extraIndices=*/{}, indexRemap,
                                 /*extraOperands=*/regionSymbols,
                                 /*symbolOperands=*/{}, userFilterFn);

  *nBegin = isBeginAtStartOfBlock ? block->begin() : std::next(prevOfBegin);

  return success();
}

/// Construct the memref region to just include the entire memref. Returns false
/// dynamic shaped memref's for now. `numParamLoopIVs` is the number of
/// enclosing loop IVs of `op` (starting from the outermost) that the region
/// is parametric on.
static bool getFullMemRefAsRegion(Operation *op, unsigned numParamLoopIVs,
                                  MemRefRegion *region) {
  unsigned rank;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    rank = loadOp.getMemRefType().getRank();
    region->memref = loadOp.getMemRef();
    region->setWrite(false);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    rank = storeOp.getMemRefType().getRank();
    region->memref = storeOp.getMemRef();
    region->setWrite(true);
  } else {
    assert(false && "expected load or store op");
    return false;
  }
  auto memRefType = cast<MemRefType>(region->memref.getType());
  if (!memRefType.hasStaticShape())
    return false;

  auto *regionCst = region->getConstraints();

  // Just get the first numSymbols IVs, which the memref region is parametric
  // on.
  SmallVector<AffineForOp, 4> ivs;
  getAffineForIVs(*op, &ivs);
  ivs.resize(numParamLoopIVs);
  SmallVector<Value, 4> symbols;
  extractForInductionVars(ivs, &symbols);
  *regionCst = FlatAffineValueConstraints(rank, numParamLoopIVs, 0);
  regionCst->setValues(rank, rank + numParamLoopIVs, symbols);

  // Memref dim sizes provide the bounds.
  for (unsigned d = 0; d < rank; d++) {
    auto dimSize = memRefType.getDimSize(d);
    assert(dimSize > 0 && "filtered dynamic shapes above");
    regionCst->addBound(BoundType::LB, d, 0);
    regionCst->addBound(BoundType::UB, d, dimSize - 1);
  }
  return true;
}

LogicalResult
