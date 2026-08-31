#include "backend_cache.hpp"
#include <llama.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace sloexp {
namespace {
std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> out; std::stringstream ss(s); std::string part;
  while (std::getline(ss,part,',')) out.push_back(part); return out;
}
std::string row(const Measurement &m) {
  return m.key+","+m.observation_id+","+m.run_id+","+std::to_string(m.timestamp_ns)+","+
    std::to_string(m.prefill_tokens)+","+std::to_string(m.decode_items)+","+
    std::to_string(m.context_tokens)+","+std::to_string(m.duration_ns)+","+
    std::to_string(m.monotonic_ns)+","+std::to_string(m.execution_position)+","+
    std::to_string(m.sequence_count)+"\n";
}
struct BackendGuard { BackendGuard(){llama_backend_init();} ~BackendGuard(){llama_backend_free();} };
void quiet_backend_log(enum ggml_log_level level, const char *text, void *) {
  if (level >= GGML_LOG_LEVEL_WARN)
    std::fputs(text, stderr);
}
struct LogGuard {
  ggml_log_callback previous{};
  void *previous_data{};
  LogGuard() {
    llama_log_get(&previous, &previous_data);
    llama_log_set(quiet_backend_log, nullptr);
  }
  ~LogGuard() { llama_log_set(previous, previous_data); }
};
struct ModelDel { void operator()(llama_model *p)const{llama_model_free(p);} };
struct ContextDel { void operator()(llama_context *p)const{llama_free(p);} };
}

std::vector<Measurement> MeasurementCache::load() const {
  std::vector<Measurement> out; std::ifstream in(path_); std::string line;
  while(std::getline(in,line)) {
    if(line=="key,observation_id,run_id,timestamp_ns,prefill_tokens,decode_items,context_tokens,duration_ns,monotonic_ns,execution_position,sequence_count") continue;
    auto f=split(line); if(f.size()!=11) throw std::runtime_error("corrupt cache row");
    try { out.push_back({f[0],f[1],f[2],std::stoull(f[3]),static_cast<std::uint32_t>(std::stoul(f[4])),
      static_cast<std::uint32_t>(std::stoul(f[5])),static_cast<std::uint32_t>(std::stoul(f[6])),std::stoull(f[7]),
      std::stoull(f[8]),static_cast<std::uint32_t>(std::stoul(f[9])),static_cast<std::uint32_t>(std::stoul(f[10]))}); }
    catch(const std::exception&){throw std::runtime_error("corrupt cache value");}
  }
  return out;
}

bool MeasurementCache::append(const Measurement &m) {
  const auto lock=std::filesystem::path(path_.string()+".lock");
  for(int tries=0; !std::filesystem::create_directory(lock); ++tries) {
    if(tries>500) throw std::runtime_error("cache lock timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  struct Unlock { std::filesystem::path p; ~Unlock(){std::error_code e;std::filesystem::remove(p,e);} } unlock{lock};
  auto existing=load();
  for(const auto &x:existing) if(x.key==m.key&&x.observation_id==m.observation_id) return false;
  std::filesystem::create_directories(path_.parent_path());
  auto tmp=std::filesystem::path(path_.string()+".tmp");
  std::ofstream out(tmp,std::ios::trunc);
  if(!out) throw std::runtime_error("cannot open cache temporary");
  out << "key,observation_id,run_id,timestamp_ns,prefill_tokens,decode_items,context_tokens,duration_ns,monotonic_ns,execution_position,sequence_count\n";
  for(const auto &x:existing) out<<row(x); out<<row(m); out.close();
  if(!out) throw std::runtime_error("cannot write cache temporary");
  std::filesystem::rename(tmp,path_); return true;
}

std::vector<std::uint64_t> measure_backend(const std::string &model_path, std::uint32_t p,
                                           std::uint32_t d, std::uint32_t context_tokens,
                                           std::uint32_t threads,
                                           std::size_t warmups, std::size_t repetitions) {
  if(p+d==0||context_tokens<2||p>context_tokens) throw std::invalid_argument("invalid batch specification");
  LogGuard log_guard;
  BackendGuard guard;
  std::unique_ptr<llama_model,ModelDel> model(llama_model_load_from_file(model_path.c_str(),llama_model_default_params()));
  if(!model) throw std::runtime_error("failed to load model");
  auto cp=llama_context_default_params(); cp.n_seq_max=std::max<std::uint32_t>(1,d+(p?1:0));
  cp.n_threads=threads;cp.n_threads_batch=threads;
  cp.n_ctx=context_tokens*cp.n_seq_max;
  cp.n_batch=std::max<std::uint32_t>(p+d,context_tokens-1);
  cp.n_ubatch=cp.n_batch;
  std::unique_ptr<llama_context,ContextDel> ctx(llama_init_from_model(model.get(),cp));
  if(!ctx) throw std::runtime_error("failed to create context");
  const llama_token token=llama_vocab_bos(llama_model_get_vocab(model.get()));
  auto run=[&](){
    llama_memory_clear(llama_get_memory(ctx.get()),true);
    for(std::uint32_t j=0;j<d;++j){
      auto seed=llama_batch_init(context_tokens-1,0,1); seed.n_tokens=context_tokens-1;
      for(std::uint32_t k=0;k<context_tokens-1;++k){seed.token[k]=token;seed.pos[k]=k;seed.n_seq_id[k]=1;seed.seq_id[k][0]=j;seed.logits[k]=0;}
      if(llama_decode(ctx.get(),seed)!=0){llama_batch_free(seed);throw std::runtime_error("decode seed failed");}
      llama_batch_free(seed);
    }
    auto batch=llama_batch_init(p+d,0,1); batch.n_tokens=p+d; std::uint32_t rowi=0;
    for(std::uint32_t k=0;k<p;++k,++rowi){batch.token[rowi]=token;batch.pos[rowi]=k;batch.n_seq_id[rowi]=1;batch.seq_id[rowi][0]=d;batch.logits[rowi]=0;}
    for(std::uint32_t j=0;j<d;++j,++rowi){batch.token[rowi]=token;batch.pos[rowi]=context_tokens-1;batch.n_seq_id[rowi]=1;batch.seq_id[rowi][0]=j;batch.logits[rowi]=0;}
    const auto a=std::chrono::steady_clock::now(); const int rc=llama_decode(ctx.get(),batch); const auto z=std::chrono::steady_clock::now(); llama_batch_free(batch);
    if(rc!=0) throw std::runtime_error("measured decode failed");
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(z-a).count());
  };
  for(std::size_t i=0;i<warmups;++i)(void)run(); std::vector<std::uint64_t> out; out.reserve(repetitions);
  for(std::size_t i=0;i<repetitions;++i)out.push_back(run()); return out;
}
} // namespace sloexp
