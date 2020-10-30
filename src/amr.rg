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

-- Point is a field space corresponding to a point on the mesh.
fspace Point {
  input: double,
  output: double,
}

-- Value is a field space corresponding to a single double.
fspace Value {
  val: double,
}

-- abs is a utility function to compute |x|.
terra abs(x: double)
  if x >= 0 then
    return x
  else
    return -x
  end
end

-- factorize constructs a rectangle of size roughly [sqrt(parallelism), sqrt(parallelism)].
task factorize(parallelism : int) : int2d
  var limit = [int](math.sqrt([double](parallelism)))
  var size_x = 1
  var size_y = parallelism
  for i = 1, limit + 1 do
    if parallelism % i == 0 then
      size_x, size_y = i, parallelism / i
      if size_x > size_y then
        size_x, size_y = size_y, size_x
      end
    end
  end
  return int2d { size_x, size_y }
end

-- initializeGrid initializes an input grid.
__demand(__cuda)
task initializeGrid(grid: region(ispace(int2d), Point))
where
  reads writes(grid)
do
  for p in grid do
    grid[p].input = COEFX * p.x + COEFY * p.y 
    grid[p].output = 0.0
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

-- applyStencil applies the statically known "star" stencil to each point
-- in the input grid. It operates on a private partition of the grid to
-- write into, and a halo partition of each private partition to read from.
__demand(__cuda)
task applyStencil(
  mult: double, 
  halo: region(ispace(int2d), Point),
  private: region(ispace(int2d), Point)
) where
  reads(halo.input, private.output),
  writes(private.output)
do
  for p in private do
    var i = p.x
    var j = p.y

    for jj=-RADIUS,0 do
      private[p].output += mult / (2.0 * RADIUS * jj) * halo[{i, j + jj}].input
    end

    for jj=1,RADIUS+1 do
      private[p].output += mult / (2.0 * RADIUS * jj) * halo[{i, j + jj}].input
    end
    
    for ii=-RADIUS,0 do
      private[p].output += mult / (2.0 * RADIUS * ii) * halo[{i+ii, j}].input
    end

    for ii=1,RADIUS+1 do
      private[p].output += mult / (2.0 * RADIUS * ii) * halo[{i+ii, j}].input
    end
  end
end

-- addConstantToInput adds a constant to every element of the input region.
__demand(__cuda)
task addConstantToInput(grid: region(ispace(int2d), Point))
where
  reads writes(grid.input)
do
  for p in grid do
    grid[p].input += 1.0
  end
end

-- Use a two-stage, bi-linear interpolation from background grid to refinement.
-- TODO (rohany): Parallelize this. It seems like it will be tricky though.
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

-- normGridOuput and normGridInput are helper functions used to collect
-- information about the resulting grids while validating the solution.
-- Note that we do not use the "abs" function defined above as that interferes
-- with CUDA code generation for these tasks.
__demand(__cuda)
task normGridOutput(grid: region(ispace(int2d), Point), numPoints: double)
where
  reads (grid)
do
  var n = 0.0
  for p in grid do
    if grid[p].output < 0 then
      n += -grid[p].output
    else
      n += grid[p].output
    end
  end
  return n / numPoints
end
__demand(__cuda)
task normGridInput(grid: region(ispace(int2d), Point), numPoints: int)
where
  reads (grid)
do
  var n = 0.0
  for p in grid do
    if grid[p].output < 0 then
      n += -grid[p].input
    else
      n += grid[p].input
    end
  end
  return n / numPoints
end

-- generateMainTask creates the main AMR task templated by
-- the number of refinement grids to consider.
function generateMainTask(N)
  -- generate creates a list of statements by mapping the range
  -- 1..n over generator.
  local function generate(n, generator)
    local stmts = terralib.newlist()
    for i = 1, n do
      stmts:insert(generator(i))
    end
    return stmts
  end
  -- generateIf creates a list of statements in a pattern of invoking
  -- the generator with value i if the passed in variable v == i - 1.
  -- This is able to simulate a dynamic branch into the set of statically
  -- constructed refinement instances.
  local function generateIf(n, v, generator)
    local stmts = terralib.newlist()
    for i = 1, n do
      stmts:insert(rquote
        if v == [i - 1] then
          [generator(i)]
        end
      end)
    end
    return stmts
  end

  -- refinements is the constant set of refinement grids.
  local refinements = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementPartitions is the constant set of equal partitions of the
  -- refinements.
  local refinementPartitions = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementInteriors is the constant set of refinement interior grids.
  local refinementInteriors = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementPrivateInteriors is the constant set of equal partitions of
  -- the refinements.
  local refinementPrivateInteriors = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementHalos is the constant set of halo partitions for each partition
  -- in refinementPrivateInteriors.
  local refinementHalos = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementHaloColorings is a set of temporary objects use to partition each
  -- refinement.
  local refinementHaloColorings = generate(4, function(i)
    return regentlib.newsymbol()
  end)
  -- refinementNorms is a set of 0's equal to the number of refinements to
  -- assist with checking of the solution.
  local refinementNorms = generate(4, function(i)
    return rexpr 0.0 end
  end)
   
  local task main()
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
    c.printf("   Parallelism            = %d\n", conf.parallelism);
    c.printf("   Refinement Parallelism = %d\n", conf.refinementParallelism);
    
    -- Create a region for the grid representing the problem.
    var gridSpace = ispace(int2d, {conf.n, conf.n})
    var gridColors = ispace(int2d, factorize(conf.parallelism))
    var grid = region(gridSpace, Point)
    var gridPartition = partition(equal, grid, gridColors)
    for c in gridColors do
      initializeGrid(gridPartition[c])
    end
    var gridInterior = createInteriorPartition(grid)[0]

    -- Partition the interior grid into equal chunks.
    var gridInteriorPrivate = partition(equal, gridInterior, gridColors)
    -- Create a halo partition for ghost region access.
    var cHalo = c.legion_domain_point_coloring_create()
    for color in gridColors do
      var bounds = gridInteriorPrivate[color].bounds
      var haloBounds = rect2d { bounds.lo - {RADIUS, RADIUS}, bounds.hi + {RADIUS, RADIUS} }
      c.legion_domain_point_coloring_color_domain(cHalo, color, haloBounds)
    end
    var gridHalo = partition(aliased, grid, cHalo, gridColors)
    c.legion_domain_point_coloring_destroy(cHalo)

    -- Set up the regions corresponding to the refinements.
    var refinementSpace = ispace(int2d, {conf.nRefinementTrue, conf.nRefinementTrue})
    var refinementColors = ispace(int2d, factorize(conf.refinementParallelism));
    [ 
      generate (4, function (i) return
        rquote 
          -- Generate assignments for each refinement.
          var [refinements[i]] = region (refinementSpace, Point)
          -- Generate equal partitions for each refinement.
          var [refinementPartitions[i]] = partition(equal, [refinements[i]], refinementColors)
          -- Create the refinement interior partitions.
          var [refinementInteriors[i]] = createInteriorPartition([refinements[i]])[0]
          -- Create the private partitions of the interior.
          var [refinementPrivateInteriors[i]] = partition(equal, [refinementInteriors[i]], refinementColors)
          -- Create a halo partition for each refinement region's ghost access.
          -- TODO (rohany): Can we refactor this code to not duplicate as much?
          var [refinementHaloColorings[i]] = c.legion_domain_point_coloring_create()
          for color in refinementColors do
            var bounds = [refinementPrivateInteriors[i]][color].bounds
            var haloBounds = rect2d { bounds.lo - {RADIUS, RADIUS}, bounds.hi + {RADIUS, RADIUS} }
            c.legion_domain_point_coloring_color_domain([refinementHaloColorings[i]], color, haloBounds)
          end
          var [refinementHalos[i]] = partition(aliased, [refinements[i]], [refinementHaloColorings[i]], refinementColors)
          c.legion_domain_point_coloring_destroy([refinementHaloColorings[i]])
          -- Initialize each refinement region.
          fill([refinements[i]].{input, output}, 0.0)
        end
      end)
    ];

    var stencilSize = 4 * RADIUS + 1
    
    -- Initialize refinement layouts.
    var istart = array(0, 0, 0, 0)
    var jstart = array(0, 0, 0, 0)
    istart[1] = conf.n - conf.refinements
    istart[3] = conf.n - conf.refinements
    jstart[1] = conf.n - conf.refinements
    jstart[2] = conf.n - conf.refinements

    -- Construct a partition of the input grid for each refinement regions.
    var rGridColors = c.legion_domain_point_coloring_create()
    var refinementGridColors = ispace(int1d, 4)
    for i in refinementGridColors do
      var offset = (conf.nRefinementTrue - 1) / conf.expand
      var idx = int(i)
      var lo = {istart[idx], jstart[idx]}
      var hi = {istart[idx] + offset, jstart[idx] + offset} 
      c.legion_domain_point_coloring_color_domain(rGridColors, i, rect2d {lo, hi})
    end
    var gridPartitionsForRefinement = partition(aliased, grid, rGridColors, refinementGridColors)
    c.legion_domain_point_coloring_destroy(rGridColors)
    
    -- Set up initial state for the main simulation loop.
    var stencilTime = 0
    var numInterpolations = 0
    var g = 0

    -- Set up fence to complete all initialization logic.
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
        -- Pick the correct refinement to interpolate.
        [
          generateIf(4, g, function(i) return rquote
              interpolateRefinement([refinements[i]], gridPartitionsForRefinement[g], conf, istart[g], jstart[g])
            end 
          end)
        ];
      end

      -- Perform any needed stencil iterations on the refinement grids.
      if (iter % conf.refinementPeriod) < conf.refinementDuration then
        var rg : region(ispace(int2d), Point)
        for subIter = 0, conf.refinementIterations do
          [
            generateIf(4, g, function(i) return
              rquote
                for color in refinementColors do
                  applyStencil(conf.expand, [refinementHalos[i]][color], [refinementPrivateInteriors[i]][color])
                end
                for color in refinementColors do
                  addConstantToInput([refinementPartitions[i]][color])
                end
              end 
            end)
          ];
        end
      end
      
      -- Apply the stencil operator to the background grid.
      for c in gridColors do
        applyStencil(1.0, gridHalo[c], gridInteriorPrivate[c])
      end
      -- Add constant to solution to force refresh of neighbor data.
      for c in gridColors do
        addConstantToInput(gridPartition[c])
      end
    end
    
    __fence(__execution, __block)
    stencilTime = c.legion_get_current_time_in_micros() - stencilTime

    var norm = 0.0
    var normIn = 0.0
    -- Compute normalized L1 solution norm on background grid.
    for c in gridColors do
      norm += normGridOutput(gridInteriorPrivate[c], conf.activePoints)
    end
    -- Compute normalized L1 input field norm on background grid.
    for c in gridColors do
      normIn += normGridInput(gridPartition[c], conf.n * conf.n)
    end

    -- Compute the same norms on each of the refinements.
    var normR = array([refinementNorms])
    var normInR = array([refinementNorms]);
    [
      generate(4, function(i) return 
        rquote
          normR[i - 1] = 0.0
          normInR[i - 1] = 0.0
          for c in refinementColors do
            -- For some reason, the code generator gets confused when this partition
            -- is referenced directly in the task call, so we pull it out into a temp.
            var p = [refinementPrivateInteriors[i]][c]
            normR[i - 1] += normGridOutput(p, conf.activeRefinementPoints)
          end
          for c in refinementColors do
            var p = [refinementPartitions[i]][c]
            normInR[i - 1] += normGridInput(p, conf.nRefinementTrue * conf.nRefinementTrue)
          end
        end
      end)
    ];

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
    -- At this point, the computation is correct!
    c.printf("Solution validates!\n")

    -- Compute the number of FLOPS.
    var flops = conf.activePoints * conf.iterations
    -- Add in the flops for each refinement.
    flops = flops * (2 * stencilSize + 1)
    iterationsR[0] -= 1
    for g = 0, 4 do
      flops += conf.activeRefinementPoints * iterationsR[0]
    end
    if conf.refinementLevel > 0 then
      numInterpolations--
      flops += conf.nRefinementTrue * (numInterpolations) * 3 * (conf.nRefinementTrue * conf.refinements)
    end
    
    -- Print out info about the run.
    var timeSeconds = stencilTime * 1e-6
    var avgTime = timeSeconds / conf.iterations
    c.printf("MFLOPS = %7.3f, Avg time (s): %7.3f\n", 1e-6 * flops / timeSeconds, timeSeconds / conf.iterations)
    c.printf("Elapsed time = %7.3f s \n", timeSeconds)
  end
  return main
end

local amrMain = generateMainTask(4)

terra registerMappers()
  -- saveobj requires a thunk to register any custom mappers.
  -- We don't have any right now, so just send in an empty thunk.
  return
end
-- TODO (rohany): If we are going to compile this on other machines
--  than Sherlock, the PMIx link should be optional.
regentlib.saveobj(amrMain, "amr", "executable", registerMappers, terralib.newlist({"-lpmix", "-lm"}))
