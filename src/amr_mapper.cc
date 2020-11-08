#include "mappers/default_mapper.h"
#include "realm/logging.h"
#include "amr_mapper.h"
#include <stdio.h>

using namespace Legion;
using namespace Legion::Mapping;

// static Realm::Logger LOG("amr_mapper");

class AMRMapper: public DefaultMapper {
public:
    AMRMapper(Runtime* rt, Machine machine, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), machine, local, "amr_mapper") {}

    // This mapper currently does nothing different than the default mapper.
    // I thought that I would need to do something by hand here after seeing
    // load balancing related problems in performance profiles, but those appear
    // to be handled by just creating more partitions. We'll keep this mapper
    // around in case we want to experiment with the LifeLine mapper though.
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
