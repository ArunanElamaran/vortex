#include "tensor_mem.h"
#include "core.h"
#include <bitmanip.h>
#include <vector>
#include "types.h"

using namespace vortex;

class TensorMem::Impl {
protected:
	TensorMem* simobject_;
	Config    config_;
	RAM       ram_;
	uint32_t 	line_bits_;
	MemCrossBar::Ptr mem_xbar_;
	mutable PerfStats perf_stats_;

	uint64_t to_local_addr(uint64_t addr) {
		// assuming addr is in [TMEM_BASE_ADDR, TMEM_BASE_ADDR + capacity)
		// TODO: CHECK IF BELOW IS APPROPRIATE CHANGE
		uint64_t local = addr - TMEM_BASE_ADDR;
		return bit_getw(local, 0, line_bits_ - 1);
	}

public:
	Impl(TensorMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
	{
		uint32_t total_lines = config.capacity / config.line_size;
		line_bits_ = log2ceil(total_lines);

		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		uint32_t lg2_line_size = log2ceil(config_.line_size);
		uint32_t num_banks = 1 << config.B;
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::Priority, config.num_reqs, num_banks,
		 [lg2_line_size, num_banks](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_line_size) & (num_banks-1));
		});
		for (uint32_t i = 0; i < config.num_reqs; ++i) {
			simobject->Inputs.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspIn.at(i).bind(&simobject->Outputs.at(i));
		}
	}

	virtual ~Impl() {}

	void reset() {
		perf_stats_ = PerfStats();
	}

	void read(void* data, uint64_t addr, uint32_t size) {
		auto l_addr = to_local_addr(addr);
		DPH(3, "Tensor Mem addr=0x" << std::hex << l_addr << std::dec << std::endl);
		ram_.read(data, l_addr, size);
	}

	void write(const void* data, uint64_t addr, uint32_t size) {
		auto l_addr = to_local_addr(addr);
		DPH(3, "Tensor Mem addr=0x" << std::hex << l_addr << std::dec << std::endl);
		ram_.write(data, l_addr, size);
	}

	void tick() {
		// process bank requets from xbar
		uint32_t num_banks = (1 << config_.B);
		for (uint32_t i = 0; i < num_banks; ++i) {
			auto& xbar_req_out = mem_xbar_->ReqOut.at(i);
			if (xbar_req_out.empty())
				continue;

			auto& bank_req = xbar_req_out.front();
			DT(4, simobject_->name() << "-bank" << i << "-req : " << bank_req);

			if (!bank_req.write || config_.write_reponse) {
				// send xbar response
				MemRsp bank_rsp{bank_req.tag, bank_req.cid, bank_req.uuid};
				mem_xbar_->RspOut.at(i).push(bank_rsp);
			}

			// update perf counters
			perf_stats_.reads += !bank_req.write;
			perf_stats_.writes += bank_req.write;

			// remove input
			xbar_req_out.pop();
		}
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		return perf_stats_;
	}
};

///////////////////////////////////////////////////////////////////////////////

TensorMem::TensorMem(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<TensorMem>(ctx, name)
	, Inputs(config.num_reqs, this)
	, Outputs(config.num_reqs, this)
	, impl_(new Impl(this, config))
{}

TensorMem::~TensorMem() {
  delete impl_;
}

void TensorMem::reset() {
  impl_->reset();
}

void TensorMem::read(void* data, uint64_t addr, uint32_t size) {
  impl_->read(data, addr, size);
}

void TensorMem::write(const void* data, uint64_t addr, uint32_t size) {
  impl_->write(data, addr, size);
}

void TensorMem::tick() {
  impl_->tick();
}

const TensorMem::PerfStats& TensorMem::perf_stats() const {
  return impl_->perf_stats();
}