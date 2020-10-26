import "regent"

require("amr_config")

-- TODO (rohany): Add a header here about what code that this was copied from.

-- Set up some local C imports.
local c = regentlib.c
local cstring = terralib.includec("string.h")
local std = terralib.includec("stdlib.h")
local math = terralib.includec("math.h")

-- Set up constants.
local EPSILON = 1e-8
local COEFX = 1.0
local COEFY = 1.0
-- TODO (rohany): Can this be a compile time flag?
local RADIUS = 2
local REFINEMENTS = 4

-- TODO (rohany): Might want a "state" struct that holds onto these things.

-- TODO (rohany): Comment these field spaces.
-- TODO (rohany): I think that this can be used for both the grid and
--  refinement regions.
fspace Point {
  input: double,
  output: double,
}

fspace Value {
  val: double,
}

terra abs(x: double)
  if x >= 0 then
    return x
  else
    return -x
  end
end

task initializeGrid(grid: region(ispace(int2d), Point))
where
  reads writes(grid)
do
  for p in grid do
    grid[p].input = COEFX * p.x + COEFY * p.y 
    grid[p].output = 0.0
  end
end

task initializeStencil(stencil: region(ispace(int2d), Value))
where
  reads writes(stencil)
do
  -- Initialize all of the stencil values.
  for p in stencil do
    stencil[p].val = 0.0
  end
  -- Fill in parts of the stencil now.
  for i = 1, RADIUS + 1 do
    var stencilVal = 1.0 / (2.0 * i * RADIUS)
    stencil[{0, i}].val = stencilVal
    stencil[{i, 0}].val = stencilVal
    stencil[{0, -i}].val = -stencilVal
    stencil[{-i, 0}].val = -stencilVal
  end
end

task initializeRefinementStencil(
  conf: Config, 
  stencil: region(ispace(int2d), Value), 
  refinement: region(ispace(int2d), Value)
) where
  reads(stencil, refinement),
  writes(refinement)
do
  for p in stencil do
    refinement[p].val = stencil[p].val * (conf.expand * 1.0)
  end
end

-- createInteriorPartition creates a partition of grid that excludes the points
-- that are closer then RADIUS to the boundaries of the grid.
task createInteriorPartition(grid: region(ispace(int2d), Point))
  var coloring = c.legion_domain_coloring_create()
  var bounds = grid.ispace.bounds
  c.legion_domain_coloring_color_domain(
    coloring,
    0,
    rect2d { bounds.lo + {RADIUS, RADIUS}, bounds.hi - {RADIUS, RADIUS} }
  )
  var interiorPartition = partition(disjoint, grid, coloring)
  c.legion_domain_coloring_destroy(coloring)
  return interiorPartition
end

-- applyStencil applies the input star stencil to each point in the input grid region.
task applyStencil(
  stencil: region(ispace(int2d), Value),
  grid: region(ispace(int2d), Point)
) where
  reads(stencil.val, grid.{input, output}),
  writes(grid.output)
do
  for p in grid do
    var i = p.x
    var j = p.y
    
    for jj=-RADIUS,RADIUS+1 do
      grid[p].output += stencil[{0, jj}].val * grid[{i, j+jj}].input
    end
    for ii=-RADIUS,0 do
      grid[p].output += stencil[{ii, 0}].val * grid[{i+ii, j}].input
    end
    for ii=1,RADIUS+1 do
      grid[p].output += stencil[{ii, 0}].val * grid[{i+ii, j}].input
    end
  end
end

-- addConstantToInput adds a constant to every element of the input region.
task addConstantToInput(grid: region(ispace(int2d), Point))
where
  reads writes(grid.input)
do
  for p in grid do
    grid[p].input += 1.0
  end
end

-- Use a two-stage, bi-linear interpolation from background grid to refinement.
task interpolateRefinement(
  refinement: region(ispace(int2d), Point),
  grid: region(ispace(int2d), Point),
  conf: Config,
  istart: int,
  jstart: int
) where
  reads(grid.{input}, refinement.{input}),
  writes(refinement.{input})
do
  -- If the same resolution, then just copy the background grid.
  if conf.expand == 1 then
    -- TODO (rohany): Should these loops be flipped?
    -- TODO (rohany): Can we change this to iteration over an index space?
    for jr = 0, conf.nRefinementTrue do
      for ir = 0, conf.nRefinementTrue do
        refinement[{ir, jr}].input = grid[{ir + istart, jr + jstart}].input
      end
    end
  else
    var iend = istart + (conf.nRefinementTrue - 1) / conf.expand
    var jend = jstart + (conf.nRefinementTrue - 1) / conf.expand

    -- First, interpolate in the x direction.
    var jb = jstart
    for jr = 0, conf.nRefinementTrue, conf.expand do
      for ir = 0, conf.nRefinementTrue - 1 do
        var xr : double = istart + conf.meshSpacing * ir
        var ib : int32 = xr
        var xb : double = ib
        refinement[{ir, jr}].input = grid[{ib+1, jb}].input * (xr - xb) + grid[{ib, jb}].input * (xb + 1.0 - xr)
      end
      refinement[{conf.nRefinementTrue - 1, jr}].input = grid[{iend, jb}].input
      jb += 1
    end

    -- Next, interpolate in the y direction.
    for jr = 0, conf.nRefinementTrue - 1 do
      var yr : double = conf.meshSpacing * jr
      var jb : int = yr
      var jrb : int = jb * conf.expand
      var jrb1 : int = (jb + 1) * conf.expand
      var yb : double = jb
      for ir = 0, conf.nRefinementTrue do
        refinement[{ir, jr}].input = refinement[{ir, jrb1}].input * (yr - yb) + refinement[{ir, jrb}].input * (yb + 1.0 - yr)
      end
    end
  end
end

task normGridOutput(grid: region(ispace(int2d), Point), numPoints: double)
where
  reads (grid)
do
  var n = 0.0
  for p in grid do
    n += abs(grid[p].output)
  end
  return n / numPoints
end

task normGridInput(grid: region(ispace(int2d), Point), numPoints: int)
where
  reads (grid)
do
  var n = 0.0
  for p in grid do
    n += abs(grid[p].input)
  end
  return n / numPoints
end


task toplevel()
  -- Set up the configuration for the problem.
  var conf : Config
  initializeConfig(&conf)
  parseInputArguments(&conf)
  validateConfig(conf)

  -- Intialize some more state in the config.
  conf.nRefinementTrue = (conf.refinements - 1) * conf.expand + 1
  conf.activePoints = (conf.n - 2 * RADIUS) * (conf.n - 2 * RADIUS)
  conf.activeRefinementPoints = (conf.nRefinementTrue - 2 * RADIUS) * (conf.nRefinementTrue - 2 * RADIUS)

  -- Print out data to match the serial code.
  c.printf("Background grid size = %d\n", conf.n)
  c.printf("Radius of stencil    = %d\n", RADIUS)
  c.printf("Type of stencil      = star\n")
  c.printf("Data type            = double precision\n")
  c.printf("Compact representation of stencil loop body\n")
  if conf.tiling then 
    c.printf("Tile size            = %d\n", conf.tileSize)
  else
    c.printf("Untiled\n")
  end
  c.printf("Number of iterations = %d\n", conf.iterations);
  c.printf("Refinements:\n");
  c.printf("   Background grid points = %ld\n", conf.refinements);
  c.printf("   Grid size              = %ld\n", conf.nRefinementTrue);
  c.printf("   Period                 = %d\n", conf.refinementPeriod);
  c.printf("   Duration               = %d\n", conf.refinementDuration);
  c.printf("   Level                  = %d\n", conf.refinementLevel);
  c.printf("   Sub-iterations         = %d\n", conf.refinementIterations);
  

  -- Create a region for the grid representing the problem.
  var gridSpace = ispace(int2d, {conf.n, conf.n})
  var grid = region(gridSpace, Point)
  initializeGrid(grid)
  var gridInterior = createInteriorPartition(grid)[0]

  -- Set up the regions corresponding to the refinements.
  var refinementSpace = ispace(int2d, {conf.nRefinementTrue, conf.nRefinementTrue})
  -- TODO (rohany): We want to replace this with some metaprogramming...
  var refinement1 = region(refinementSpace, Point)
  var refinement2 = region(refinementSpace, Point)
  var refinement3 = region(refinementSpace, Point)
  var refinement4 = region(refinementSpace, Point)
  var refinementInterior1 = createInteriorPartition(refinement1)[0]
  var refinementInterior2 = createInteriorPartition(refinement2)[0]
  var refinementInterior3 = createInteriorPartition(refinement3)[0]
  var refinementInterior4 = createInteriorPartition(refinement4)[0]
  -- Initialize the refinements with 0.
  fill(refinement1.{input, output}, 0.0)
  fill(refinement2.{input, output}, 0.0)
  fill(refinement3.{input, output}, 0.0)
  fill(refinement4.{input, output}, 0.0)

  -- Create a region for the stencil. We offset the index space to get an index
  -- space from {-RADIUS, -RADIUS} to {RADIUS, RADIUS}.
  var stencilSize = 4 * RADIUS + 1
  var stencilSpace = ispace(int2d, {2 * RADIUS + 1, 2 * RADIUS + 1}, {-1 * RADIUS, -1 * RADIUS})
  var stencil = region(stencilSpace, Value)
  var refinementStencil = region(stencilSpace, Value)
  initializeStencil(stencil)
  initializeRefinementStencil(conf, stencil, refinementStencil)

  -- Initialize refinement layouts.
  var istart = array(0, 0, 0, 0)
  var jstart = array(0, 0, 0, 0)
  istart[1] = conf.n - conf.refinements
  istart[3] = conf.n - conf.refinements
  jstart[1] = conf.n - conf.refinements
  jstart[2] = conf.n - conf.refinements
  
  var stencilTime = 0
  var numInterpolations = 0
  var g = 0
 
  __fence(__execution, __block)

  for iter = 0, conf.iterations + 1 do
    -- Start timer after a warmup iteration.
    if iter == 1 then
      stencilTime = c.legion_get_current_time_in_micros()
    end

    -- If we're at an iteration where a refinement should occur, interpolate
    -- to create the refinement.
    if iter % conf.refinementPeriod == 0 then
      g = (iter / conf.refinementPeriod) % 4
      numInterpolations += 1
      if g == 0 then
        interpolateRefinement(refinement1, grid, conf, istart[g], jstart[g])
      elseif g == 1 then
        interpolateRefinement(refinement2, grid, conf, istart[g], jstart[g])
      elseif g == 2 then
        interpolateRefinement(refinement3, grid, conf, istart[g], jstart[g])
      else
        interpolateRefinement(refinement4, grid, conf, istart[g], jstart[g])
      end
    end
    
    -- Perform any needed stencil iterations on the refinement grids.
    if (iter % conf.refinementPeriod) < conf.refinementDuration then
      var rg : region(ispace(int2d), Point)
      for subIter = 0, conf.refinementIterations do
        if g == 0 then
          applyStencil(refinementStencil, refinementInterior1)
        elseif g == 1 then
          applyStencil(refinementStencil, refinementInterior2)
        elseif g == 2 then
          applyStencil(refinementStencil, refinementInterior3)
        else
          applyStencil(refinementStencil, refinementInterior4)
        end
        if g == 0 then
          addConstantToInput(refinement1)
        elseif g == 1 then
          addConstantToInput(refinement2)
        elseif g == 2 then
          addConstantToInput(refinement3)
        else
          addConstantToInput(refinement4)
        end
      end
    end

    -- Apply the stencil operator to the background grid.
    applyStencil(stencil, gridInterior)
    -- Add constant to solution to force refresh of neighbor data.
    addConstantToInput(grid)
  end

  __fence(__execution, __block)
  stencilTime = c.legion_get_current_time_in_micros() - stencilTime

  -- Compute normalized L1 solution norm on background grid.
  var norm : double = normGridOutput(gridInterior, conf.activePoints)
  -- Compute normalized L1 input field norm on background grid.
  var normIn : double = normGridInput(grid, conf.n * conf.n)

  -- Compute the same norms on each of the refinements.
  var normR = array(0.0, 0.0, 0.0, 0.0)
  var normInR = array(0.0, 0.0, 0.0, 0.0)
  normR[0] = normGridOutput(refinementInterior1, conf.activeRefinementPoints)
  normInR[0] = normGridInput(refinement1, conf.nRefinementTrue * conf.nRefinementTrue)
  normR[1] = normGridOutput(refinementInterior2, conf.activeRefinementPoints)
  normInR[1] = normGridInput(refinement2, conf.nRefinementTrue * conf.nRefinementTrue)
  normR[2] = normGridOutput(refinementInterior3, conf.activeRefinementPoints)
  normInR[2] = normGridInput(refinement3, conf.nRefinementTrue * conf.nRefinementTrue)
  normR[3] = normGridOutput(refinementInterior4, conf.activeRefinementPoints)
  normInR[3] = normGridInput(refinement4, conf.nRefinementTrue * conf.nRefinementTrue)

  -- Verify correctness of background grid solution and input field.
  var referenceNorm : double = (conf.iterations + 1) * (COEFX + COEFY);
  var referenceNormIn : double = (COEFX + COEFY) * (1.0) * ((conf.n - 1)/2.0) + conf.iterations + 1
  regentlib.assert(abs(norm - referenceNorm) <= EPSILON, "computed norm too large")
  regentlib.assert(abs(normIn - referenceNormIn) <= EPSILON, "computed input norm too large")

  -- Verify correctness of the refinement grid.
  var fullCycles = ((conf.iterations + 1) / (conf.refinementPeriod * 4))
  var leftoverIterations = (conf.iterations + 1) % (conf.refinementPeriod * 4)
  var referenceNormR = array(0.0, 0.0, 0.0, 0.0)
  var referenceNormInR = array(0.0, 0.0, 0.0, 0.0)
  var iterationsR = array(0.0, 0.0, 0.0, 0.0)
  for g = 0, 4 do
    iterationsR[g] = conf.refinementIterations * (fullCycles * conf.refinementDuration + min(max(0, leftoverIterations - g * conf.refinementPeriod), conf.refinementDuration))
    referenceNormR[g] = iterationsR[g] * (COEFX + COEFY)
    if iterationsR[g] == 0 then
      referenceNormInR[g] = 0
    else
      var bgUpdates = (fullCycles * 4 + g) * conf.refinementPeriod
      var rUpdates = min(max(0, leftoverIterations - g * conf.refinementPeriod), conf.refinementDuration) * conf.refinementIterations
      if bgUpdates > conf.iterations then
        bgUpdates -= 4 * conf.refinementPeriod
        rUpdates = conf.refinementIterations * conf.refinementDuration
      end
      referenceNormInR[g] = (COEFX * istart[g] + COEFY * jstart[g]) + (COEFX + COEFY) * (conf.refinements - 1) / 2.0 + bgUpdates + rUpdates
    end
    regentlib.assert(abs(normR[g] - referenceNormR[g]) <= EPSILON, "computed refinement norm too large")
    regentlib.assert(abs(normInR[g] - referenceNormInR[g]) <= EPSILON, "computed refinement input norm too large")
  end

  c.printf("Solution validates!\n")

  -- TODO (rohany): Compute the number of FLOPS.
  var flops = conf.activePoints * conf.iterations
  -- TODO (rohany): Add in the flops for each refinement.
  flops = flops * (2 * stencilSize + 1)
  if conf.refinementLevel > 0 then
    numInterpolations--
    flops += conf.nRefinementTrue * (numInterpolations) * 3 * (conf.nRefinementTrue * conf.refinements)
  end
  -- TODO (rohany): Display the number of flops.

  c.printf("Elapsed time = %7.3f s \n", stencilTime * 1e-6)
end

terra registerMappers()
  -- saveobj requires a thunk to register any custom mappers.
  -- We don't have any right now, so just send in an empty thunk.
  return
end
-- TODO (rohany): If we are going to compile this on other machines
--  than Sherlock, the PMIx link should be optional.
regentlib.saveobj(toplevel, "amr", "executable", registerMappers, terralib.newlist({"-lpmix"}))
