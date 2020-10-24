import "regent"

-- Set up some local C imports.
local c = regentlib.c
local cstring = terralib.includec("string.h")
local std = terralib.includec("stdlib.h")

-- Struct describing the configuration of the AMR simulation.
struct Config {
  -- Command line parameters.
  iterations: int,
  n : int,
  refinements: int,
  refinementLevel: int,
  refinementPeriod: int,
  refinementDuration: int,
  refinementIterations: int,
  tiling: bool,
  tileSize: int,

  -- Scratch values.
  meshSpacing: double,
  expand: int, -- number of refinement cells per background cell 

  nRefinementTrue: int,
  
  activePoints: double,
  activeRefinementPoints: double,
}

-- Initialize a config with zero values.
terra initializeConfig(conf: &Config)
  conf.iterations = 0
  conf.n = 0
  conf.refinements = 0
  conf.refinementLevel = 0
  conf.refinementDuration = 0
  conf.refinementPeriod = 0
  conf.refinementIterations = 0
  conf.tiling = false
  conf.tileSize = 0

  conf.meshSpacing = 1.0
  conf.expand = 1
end

-- Parse command line input flags into a config.
terra parseInputArguments(conf: &Config)
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-i") == 0 then
      i = i + 1
      conf.iterations = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-n") == 0 then
      i = i + 1
      conf.n = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-r") == 0 then
      i = i + 1
      conf.refinements = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-rl") == 0 then
      i = i + 1
      conf.refinementLevel = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-rp") == 0 then
      i = i + 1
      conf.refinementPeriod = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-rd") == 0 then
      i = i + 1
      conf.refinementDuration = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-ri") == 0 then
      i = i + 1
      conf.refinementIterations = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-t") == 0 then
      i = i + 1
      conf.tileSize = std.atoi(args.argv[i])
      conf.tiling = true
    end
  end

  conf.meshSpacing = 1.0
  conf.expand = 1
  for l = 0, conf.refinementLevel do
    conf.meshSpacing = conf.meshSpacing / 2.0
    conf.expand = conf.expand * 2
  end
end

-- Validate a parsed config.
terra validateConfig(conf: Config)
  regentlib.assert(conf.iterations >= 1, "iterations must be >= 1")
  regentlib.assert(conf.n >= 2, "grid must have at least one cell")
  regentlib.assert(conf.refinements >= 2, "refinements must have at least once cell")
  regentlib.assert(conf.refinements <= conf.n, "refinements must be contained in background grid")
  regentlib.assert(conf.refinementLevel >= 0, "refinement levels must be >= 0")
  regentlib.assert(conf.refinementPeriod >= 1, "refinement period must be at least one")
  regentlib.assert(conf.refinementDuration >= 1 and conf.refinementDuration <= conf.refinementPeriod, "refinement duration must be between 1 and refinement period")
  regentlib.assert(conf.refinementIterations >= 1, "refinement subiterations must be positive")
  if conf.tiling then
    regentlib.assert(conf.tileSize > 0 and conf.tileSize <= conf.n, "tileSize must be > 0 and <= n")
  end
end

-- Pretty print a config.
terra printConfig(conf: Config)
  c.printf("Iterations: %d\n", conf.iterations)
  c.printf("Refinements:\n")
  c.printf("\tGrid Size: %d\n", conf.n)
  c.printf("\t Refinement number: %d\n", conf.refinements)
  c.printf("\t Refinement level: %d\n", conf.refinementLevel)
  c.printf("\t Refinement period: %d\n", conf.refinementPeriod)
  c.printf("\t Refinement duration: %d\n", conf.refinementDuration)
  c.printf("\t Refinement iterations: %d\n", conf.refinementIterations)
  if conf.tiling then
    c.printf("\t Tile size: %d\n", conf.tileSize)
  end
end
