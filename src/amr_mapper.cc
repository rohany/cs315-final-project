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
