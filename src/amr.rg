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

task toplevel()
  -- Set up the configuration for the problem.
  var conf : Config
  initializeConfig(&conf)
  parseInputArguments(&conf)
  validateConfig(conf)
  printConfig(conf)

  -- Intialize some more state in the config.
  conf.nRefinementTrue = (conf.refinements - 1) * conf.expand + 1
  conf.activePoints = (conf.n - 2 * RADIUS) * (conf.n - 2 * RADIUS)
  conf.activeRefinementPoints = (conf.nRefinementTrue - 2 * RADIUS) * (conf.nRefinementTrue - 2 * RADIUS)

  -- Regions that we want:
  -- There are input and output regions of size n*n.
  -- There are input and output regions for each of the refinements?
  -- There is a stencil region that has some weights in it I think.

  -- Create a region for the grid representing the problem.
  var gridSpace = ispace(int2d, {conf.n, conf.n})
  var grid = region(gridSpace, Point)
  initializeGrid(grid)
  var gridInterior = createInteriorPartition(grid)[0]

  -- TODO (rohany): The refinements are stored in this chunk of 4 grid size refinements. Whats up with that?
  -- TODO (rohany): Have to initialize the refinement regions.

  -- Create a region for the stencil. We offset the index space to get an index
  -- space from {-RADIUS, -RADIUS} to {RADIUS, RADIUS}.
  var stencilSize = 4 * RADIUS + 1
  var stencilSpace = ispace(int2d, {2 * RADIUS + 1, 2 * RADIUS + 1}, {-1 * RADIUS, -1 * RADIUS})
  var stencil = region(stencilSpace, Value)
  var refinementStencil = region(stencilSpace, Value)
  initializeStencil(stencil)
  initializeRefinementStencil(conf, stencil, refinementStencil)
  

  -- TODO (rohany): Initialize refinement arrays.

  __fence(__execution, __block)

  var stencilTime = 0
  var numInterpolations = 0
 
  -- TODO (rohany): Main AMR operation.

  for iter = 0, conf.iterations + 1 do
    -- Start timer after a warmup iteration.
    if iter == 1 then
      stencilTime = c.legion_get_current_time_in_micros()
    end

    -- TODO (rohany): Interpolate the refinements into life.
    
    -- TODO (rohany): Maybe perform sub iterations?

    -- Apply the stencil operator to the background grid.
    applyStencil(stencil, gridInterior)
    -- Add constant to solution to force refresh of neighbor data.
    addConstantToInput(grid)
  end

  __fence(__execution, __block)
  stencilTime = c.legion_get_current_time_in_micros() - stencilTime

  -- TODO (rohany): Compute normalized L1 solution norm on background grid.

  -- TODO (rohany): Compute normalized L1 input field norm on background grid.

  -- TODO (rohany): Compute the same norms on each of the refinements.

  -- TODO (rohany): Verify correctness of background grid solution and input field.
  var referenceNorm = (conf.iterations + 1) * (COEFX + COEFY);
  var referenceNormIn = (COEFX + COEFY) * (1.0) * ((conf.n - 1)/2.0) + conf.iterations + 1
  -- TODO (rohany): Uncomment once we have computed the norm.
  -- regentlib.assert(math.abs(norm - referenceNorm) <= EPSILON, "computed norm too large")
  -- regentlib.assert(math.abs(norm_in - referenceNormIn) <= EPSILON, "computed input norm too large")

  -- TODO (rohany): Verify correctness of the refinement grid.
  var fullCycles = ((conf.iterations + 1) / (conf.refinementPeriod * 4))
  var leftoverIterations = (conf.iterations + 1) % (conf.refinementPeriod * 4)

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
