#include "mappers/default_mapper.h"
#include "realm/logging.h"
#include "amr_mapper.h"
#include <stdio.h>
#include <iostream>

using namespace Legion;
using namespace Legion::Mapping;

class AMRMapper: public DefaultMapper {
public:
    AMRMapper(Runtime* rt, Machine machine, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), machine, local, "amr_mapper") {}

    // slice_task overrides the DefaultMapper's implementation of slice_task. We do this
    // to change the way that tasks are physically allocated on different nodes.
    void slice_task(const MapperContext      ctx,
                    const Task&              task, 
                    const SliceTaskInput&    input,
                          SliceTaskOutput&   output) {
      if (strcmp(task.get_task_name(), "applyStencil") == 0 || 
          strcmp(task.get_task_name(), "applyRefinementStencil") == 0 || 
          strcmp(task.get_task_name(), "addConstantToInput") == 0) {
        assert(input.domain.get_dim() == 2);
        // Check if we have this information cached already.
        auto finder = this->cached_slices[task.task_id].find(input.domain);
        if (finder != this->cached_slices[task.task_id].end()) {
          output.slices = finder->second;
          return;
        }
        // We're going to attempt to block the index space into chunks equal to
        // the number of nodes in the machine.
        DomainT<2,coord_t> point_space = input.domain;
        Point<2, coord_t> num_blocks = this->default_select_num_blocks<2>(this->remote_cpus.size(), point_space.bounds);
        this->decompose_points<2>(point_space, num_blocks, output.slices);
        // Store the result into our cache.
        this->cached_slices[task.task_id][input.domain] = output.slices;
      } else {
        DefaultMapper::slice_task(ctx, task, input, output);
      }
    }

  // decompose_points assigns a processor for each point in the input domain.
  template<int DIM>
  void decompose_points(const DomainT<DIM, coord_t> &point_space, const Point<DIM, coord_t>& num_blocks, std::vector<TaskSlice> &slices) {
    Point<DIM,coord_t> zeroes;
    for (int i = 0; i < DIM; i++)
        zeroes[i] = 0;
    Point<DIM,coord_t> ones;
    for (int i = 0; i < DIM; i++)
        ones[i] = 1;
    Point<DIM,coord_t> num_points =
            point_space.bounds.hi - point_space.bounds.lo + ones;
    // Create a rectangle that has a point for each block in the domain.
    Rect<DIM,coord_t> blocks(zeroes, num_blocks - ones);
    slices.reserve(blocks.volume());
    size_t node = 0;
    // We'll allocate each node a block.
    for (PointInRectIterator<DIM> pir(blocks); pir(); pir++) {
      // Construct the range of points to allocate to this block.
      Point<DIM,coord_t> block_lo = *pir;
      Point<DIM,coord_t> block_hi = *pir + ones;
      Point<DIM,coord_t> slice_lo = num_points * block_lo / num_blocks + point_space.bounds.lo;
      Point<DIM,coord_t> slice_hi = num_points * block_hi / num_blocks + point_space.bounds.lo - ones;

      // Create a rectangle of points within each chunk to iterate over.
      Rect<DIM, coord_t> chunk(slice_lo, slice_hi);
      // Get all available processors on the current machine.
      Machine::ProcessorQuery all(this->machine);
      if (this->local_gpus.size() > 0) {
        all.only_kind(Processor::TOC_PROC);
      } else {
        all.only_kind(Processor::LOC_PROC);
      }
      all.same_address_space_as(this->remote_cpus[node++ % this->remote_cpus.size()]);
      std::vector<Processor> targets(all.begin(), all.end());
      size_t next_index = 0;
      // Assign each point in this chunk to one of the available processors on this machine.
      for (PointInRectIterator<DIM> pic(chunk); pic(); pic++) {
        DomainT<DIM,coord_t> slice_space;
        slice_space.bounds.lo = *pic;
        slice_space.bounds.hi = *pic;
        slice_space.sparsity = point_space.sparsity;
        if (!slice_space.dense())
          slice_space = slice_space.tighten();
        if (slice_space.volume() > 0) {
          TaskSlice slice;
          slice.domain = slice_space;
          slice.proc = targets[next_index++ % targets.size()];
          slice.recurse = false;
          slice.stealable = false;
          slices.push_back(slice);
        }
      }
    }
  }
private:
  std::map<TaskID, std::map<Domain, std::vector<TaskSlice>>> cached_slices;
};

static void create_mappers(Machine machine,
                           Runtime* rt,
                           const std::set<Processor>& local_procs) {
    for (Processor proc : local_procs) {
        rt->replace_default_mapper(new AMRMapper(rt, machine, proc), proc);
    }
}

void register_mappers() {
    Runtime::add_registration_callback(create_mappers);
}
