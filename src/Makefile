REGENT=${CS315_REGENT}/regent.py
CXXFLAGS=-std=c++11 -g -O2 -Wall -Werror -fno-strict-aliasing -I$(CS315_REGENT)/../runtime -I$(CS315_REGENT)/../bindings/regent
LINK_FLAGS=-lm -lpmix -L$(CS315_REGENT)/../bindings/regent -lregent
# For some reason, I can't use the Realm::Logger unless I use this link path,
# rather than the build for CS315.
# LINK_FLAGS=-lm -lpmix -L$(LEGION_DIR)/bindings/regent -lregent
export INCLUDE_PATH := .

amr: amr_mapper.o amr.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

amr.o: amr.rg amr_config.rg
	$(REGENT) $^

amr_mapper.o: amr_mapper.cc amr_mapper.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f amr *.o
