void AffineDataCopyGeneration::runOnOperation() {
  func::FuncOp f = getOperation();
  OpBuilder topBuilder(f.getBody());
  zeroIndex = arith::ConstantIndexOp::create(topBuilder, f.getLoc(), 0);

  // Nests that are copy-in's or copy-out's; the root AffineForOps of those
  // nests are stored herein.
  DenseSet<Operation *> copyNests;

  for (auto &block : f)
    runOnBlock(&block, copyNests);

  // Promote any single iteration loops in the copy nests and collect
  // load/stores to simplify.
  SmallVector<Operation *, 4> copyOps;
  for (Operation *nest : copyNests)
    // With a post order walk, the erasure of loops does not affect
    // continuation of the walk or the collection of load/store ops.
    nest->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        (void)promoteIfSingleIteration(forOp);
      else if (isa<AffineLoadOp, AffineStoreOp>(op))
        copyOps.push_back(op);
    });

  // Promoting single iteration loops could lead to simplification of
  // contained load's/store's, and the latter could anyway also be
  // canonicalized.
  RewritePatternSet patterns(&getContext());
  AffineLoadOp::getCanonicalizationPatterns(patterns, &getContext());
  AffineStoreOp::getCanonicalizationPatterns(patterns, &getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyOpPatternsGreedily(
      copyOps, frozenPatterns,
      GreedyRewriteConfig().setStrictness(
          GreedyRewriteStrictness::ExistingAndNewOps));
}
