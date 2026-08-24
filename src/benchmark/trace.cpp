#include "benchmark/trace.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace quickserve::benchmark {
namespace {
constexpr std::size_t header_size = 96;
constexpr std::size_t record_size = 16;
constexpr std::array<unsigned char, 8> magic{'Q','S','T','R','A','C','E',0};

class Sha256 {
public:
  void update(const unsigned char *data, std::size_t length) {
    bit_length_ += static_cast<std::uint64_t>(length) * 8;
    while (length) {
      const auto take = std::min(length, block_.size() - used_);
      std::copy_n(data, take, block_.begin() + static_cast<std::ptrdiff_t>(used_));
      used_ += take; data += take; length -= take;
      if (used_ == block_.size()) { transform(); used_ = 0; }
    }
  }
  std::array<unsigned char, 32> finish() {
    block_[used_++] = 0x80;
    if (used_ > 56) { std::fill(block_.begin() + used_, block_.end(), 0); transform(); used_ = 0; }
    std::fill(block_.begin() + used_, block_.begin() + 56, 0);
    for (int i = 0; i < 8; ++i) block_[63-i] = static_cast<unsigned char>(bit_length_ >> (i*8));
    transform();
    std::array<unsigned char, 32> out{};
    for (std::size_t i = 0; i < state_.size(); ++i)
      for (int j = 0; j < 4; ++j) out[i*4+j] = static_cast<unsigned char>(state_[i] >> (24-j*8));
    return out;
  }
private:
  static std::uint32_t rotr(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
  void transform() {
    static constexpr std::array<std::uint32_t, 64> k{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,64> w{};
    for (int i=0;i<16;++i) w[i]=(std::uint32_t(block_[i*4])<<24)|(std::uint32_t(block_[i*4+1])<<16)|(std::uint32_t(block_[i*4+2])<<8)|block_[i*4+3];
    for (int i=16;i<64;++i) { auto s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3); auto s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for(int i=0;i<64;++i){auto s1=rotr(e,6)^rotr(e,11)^rotr(e,25);auto ch=(e&f)^((~e)&g);auto t1=h+s1+ch+k[i]+w[i];auto s0=rotr(a,2)^rotr(a,13)^rotr(a,22);auto maj=(a&b)^(a&c)^(b&c);auto t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
  std::array<std::uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::array<unsigned char,64> block_{}; std::size_t used_{}; std::uint64_t bit_length_{};
};

std::string hex(const std::array<unsigned char,32> &digest) { std::ostringstream s; s<<std::hex<<std::setfill('0'); for(auto b:digest)s<<std::setw(2)<<unsigned(b); return s.str(); }
template<class T> void put_le(std::ostream &out,T v){ using U=std::make_unsigned_t<T>;U u=static_cast<U>(v);for(std::size_t i=0;i<sizeof(T);++i)out.put(static_cast<char>(u>>(i*8))); }
template<class T> T get_le(std::istream &in){using U=std::make_unsigned_t<T>;U u{};for(std::size_t i=0;i<sizeof(T);++i){int c=in.get();if(c==EOF)throw std::runtime_error("truncated trace");u|=U(static_cast<unsigned char>(c))<<(i*8);}return static_cast<T>(u);}

bool hashed_getline(std::ifstream &in, std::string &line, Sha256 &hash) {
  if (!std::getline(in, line)) {
    if (!in.eof()) throw std::runtime_error("failed reading source CSV");
    return false;
  }
  hash.update(reinterpret_cast<const unsigned char *>(line.data()), line.size());
  if (!in.eof()) {
    static constexpr unsigned char newline = '\n';
    hash.update(&newline, 1);
  }
  return true;
}

std::int64_t days_from_civil(int y,unsigned m,unsigned d){y-=m<=2;const int era=(y>=0?y:y-399)/400;const unsigned yoe=static_cast<unsigned>(y-era*400);const unsigned doy=(153*(m+(m>2?-3:9))+2)/5+d-1;const unsigned doe=yoe*365+yoe/4-yoe/100+doy;return static_cast<std::int64_t>(era)*146097+doe-719468;}
bool leap(int y){return y%4==0&&(y%100!=0||y%400==0);}
int parse_digits(std::string_view s,std::size_t pos,std::size_t n){if(pos+n>s.size())throw std::runtime_error("invalid timestamp");int v=0;for(std::size_t i=0;i<n;++i){if(s[pos+i]<'0'||s[pos+i]>'9')throw std::runtime_error("invalid timestamp");v=v*10+s[pos+i]-'0';}return v;}
std::int64_t parse_timestamp(std::string_view s){
  if(s.size()<20||s[4]!='-'||s[7]!='-'||s[10]!=' '||s[13]!=':'||s[16]!=':')throw std::runtime_error("invalid timestamp");
  int y=parse_digits(s,0,4),mo=parse_digits(s,5,2),d=parse_digits(s,8,2),hh=parse_digits(s,11,2),mm=parse_digits(s,14,2),ss=parse_digits(s,17,2);std::size_t p=19;std::int64_t frac=0;
  if(mo<1||mo>12||d<1||d>std::array<int,12>{31,28+(leap(y)?1:0),31,30,31,30,31,31,30,31,30,31}[mo-1]||hh>23||mm>59||ss>59)throw std::runtime_error("invalid timestamp");
  if(p<s.size()&&s[p]=='.'){++p;auto start=p;while(p<s.size()&&s[p]>='0'&&s[p]<='9'&&p-start<9){frac=frac*10+(s[p++]-'0');}auto digits=p-start;if(digits==0||(p<s.size()&&s[p]>='0'&&s[p]<='9'))throw std::runtime_error("invalid timestamp precision");while(digits++<9)frac*=10;}
  int offset=0;if(p<s.size()&&s[p]=='Z'){++p;}else if(p+6==s.size()&&(s[p]=='+'||s[p]=='-')&&s[p+3]==':'){int oh=parse_digits(s,p+1,2),om=parse_digits(s,p+4,2);if(oh>23||om>59)throw std::runtime_error("invalid timezone");offset=(oh*60+om)*60*(s[p]=='+'?1:-1);p+=6;}else throw std::runtime_error("timestamp timezone required");if(p!=s.size())throw std::runtime_error("invalid timestamp");
  __int128 seconds=static_cast<__int128>(days_from_civil(y,mo,d))*86400+hh*3600+mm*60+ss-offset;__int128 ns=seconds*1000000000+frac;if(ns<std::numeric_limits<std::int64_t>::min()||ns>std::numeric_limits<std::int64_t>::max())throw std::runtime_error("timestamp out of range");return static_cast<std::int64_t>(ns);
}
std::uint32_t parse_count(std::string_view s){std::uint64_t v{};auto [ptr,ec]=std::from_chars(s.data(),s.data()+s.size(),v);if(ec!=std::errc{}||ptr!=s.data()+s.size()||v==0||v>std::numeric_limits<std::uint32_t>::max())throw std::runtime_error("invalid token count");return static_cast<std::uint32_t>(v);}
std::array<std::string_view,3> fields(const std::string &line){std::array<std::string_view,3> f{};std::size_t a=0;for(int i=0;i<3;++i){auto b=line.find(',',a);if(i<2&&b==std::string::npos)throw std::runtime_error("invalid CSV row");if(i==2&&b!=std::string::npos)throw std::runtime_error("invalid CSV row");f[i]=std::string_view(line).substr(a,b==std::string::npos?line.size()-a:b-a);a=b+1;}return f;}
void publish_exclusive(const std::filesystem::path &tmp,const std::filesystem::path &dst){
#if defined(__APPLE__)
  if(renamex_np(tmp.c_str(),dst.c_str(),RENAME_EXCL)!=0)throw std::system_error(errno,std::generic_category(),"cannot publish trace");
#elif defined(__linux__) && defined(SYS_renameat2)
  if(syscall(SYS_renameat2,AT_FDCWD,tmp.c_str(),AT_FDCWD,dst.c_str(),1)!=0)throw std::system_error(errno,std::generic_category(),"cannot publish trace");
#else
  if(std::filesystem::exists(dst))throw std::runtime_error("destination exists");std::filesystem::rename(tmp,dst);
#endif
}
} // namespace

std::string sha256_hex(const std::string &bytes){Sha256 h;h.update(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size());return hex(h.finish());}

void prepare_trace(const std::filesystem::path &csv,const std::filesystem::path &dst){
  if(std::filesystem::exists(dst))throw std::runtime_error("destination exists");
  auto tmp=dst;tmp += ".tmp."+std::to_string(::getpid())+"."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  int fd=::open(tmp.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::system_error(errno,std::generic_category(),"cannot create temporary trace");::close(fd);
  try {
    std::fstream out(tmp,std::ios::binary|std::ios::in|std::ios::out);if(!out)throw std::runtime_error("cannot open temporary trace");std::array<char,header_size> zero{};out.write(zero.data(),zero.size());
    std::ifstream in(csv,std::ios::binary);if(!in)throw std::runtime_error("cannot open source CSV");Sha256 source_hash;std::string line;if(!hashed_getline(in,line,source_hash))throw std::runtime_error("empty CSV");if(!line.empty()&&line.back()=='\r')line.pop_back();if(line!="TIMESTAMP,ContextTokens,GeneratedTokens")throw std::runtime_error("invalid CSV schema");
    std::uint64_t count=0;std::int64_t first=0,last=0;while(hashed_getline(in,line,source_hash)){if(!line.empty()&&line.back()=='\r')line.pop_back();if(line.empty())throw std::runtime_error("empty CSV row");auto f=fields(line);auto ts=parse_timestamp(f[0]);auto context=parse_count(f[1]);auto generated=parse_count(f[2]);if(count&&ts<last)throw std::runtime_error("timestamps decrease");if(!count)first=ts;last=ts;auto delta=static_cast<__int128>(ts)-first;if(delta<0||delta>std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("arrival offset overflow");put_le(out,static_cast<std::uint64_t>(delta));put_le(out,context);put_le(out,generated);if(count==std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("record count overflow");++count;}
    if(count==0)throw std::runtime_error("empty trace");const auto digest=source_hash.finish();out.seekp(0);out.write(reinterpret_cast<const char*>(magic.data()),magic.size());put_le(out,std::uint32_t{1});put_le(out,std::uint32_t{header_size});put_le(out,std::uint32_t{record_size});put_le(out,std::uint32_t{0});put_le(out,count);put_le(out,first);put_le(out,last);out.write(reinterpret_cast<const char*>(digest.data()),digest.size());std::array<char,16> reserved{};out.write(reserved.data(),reserved.size());out.flush();if(!out)throw std::runtime_error("failed writing trace");out.close();
    fd=::open(tmp.c_str(),O_RDONLY);if(fd<0||::fsync(fd)!=0){if(fd>=0)::close(fd);throw std::runtime_error("failed syncing trace");}::close(fd);publish_exclusive(tmp,dst);
  } catch (...) { std::error_code ec;std::filesystem::remove(tmp,ec);throw; }
}

TraceReader::TraceReader(std::filesystem::path path):path_(std::move(path)){
  std::ifstream in(path_,std::ios::binary);if(!in)throw std::runtime_error("cannot open trace");std::array<unsigned char,8> got{};in.read(reinterpret_cast<char*>(got.data()),got.size());if(got!=magic)throw std::runtime_error("invalid trace magic");if(get_le<std::uint32_t>(in)!=1||get_le<std::uint32_t>(in)!=header_size||get_le<std::uint32_t>(in)!=record_size||get_le<std::uint32_t>(in)!=0)throw std::runtime_error("unsupported trace header");header_.record_count=get_le<std::uint64_t>(in);header_.first_timestamp_ns=get_le<std::int64_t>(in);header_.last_timestamp_ns=get_le<std::int64_t>(in);std::array<unsigned char,32>d{};in.read(reinterpret_cast<char*>(d.data()),d.size());header_.source_sha256_hex=hex(d);std::array<unsigned char,16>r{};in.read(reinterpret_cast<char*>(r.data()),r.size());if(!in||std::any_of(r.begin(),r.end(),[](auto x){return x!=0;}))throw std::runtime_error("invalid reserved bytes");if(header_.record_count==0)throw std::runtime_error("empty trace");if(header_.last_timestamp_ns<header_.first_timestamp_ns)throw std::runtime_error("trace timestamps decrease");if(header_.record_count>(std::numeric_limits<std::uintmax_t>::max()-header_size)/record_size||std::filesystem::file_size(path_)!=header_size+header_.record_count*record_size)throw std::runtime_error("trace size mismatch");
}
TraceRecord TraceReader::record(std::uint64_t index)const{if(index>=header_.record_count)throw std::out_of_range("trace record index");std::ifstream in(path_,std::ios::binary);in.seekg(static_cast<std::streamoff>(header_size+index*record_size));return {get_le<std::uint64_t>(in),get_le<std::uint32_t>(in),get_le<std::uint32_t>(in)};}
TraceCursor TraceReader::cursor() const { return TraceCursor(path_, header_); }

TraceCursor::TraceCursor(const std::filesystem::path &path,
                         const TraceHeader &header)
    : input_(path, std::ios::binary), header_(header) {
  if (!input_) throw std::runtime_error("cannot open trace");
  input_.seekg(static_cast<std::streamoff>(header_size));
  if (!input_) throw std::runtime_error("cannot seek trace records");
}

bool TraceCursor::next(TraceRecord &record) {
  if (index_ == header_.record_count) return false;
  record = {get_le<std::uint64_t>(input_), get_le<std::uint32_t>(input_),
            get_le<std::uint32_t>(input_)};
  if ((index_ == 0 && record.arrival_offset_ns != 0) ||
      (index_ != 0 && record.arrival_offset_ns < previous_offset_) ||
      record.context_tokens == 0 || record.generated_tokens == 0) {
    throw std::runtime_error("invalid trace record");
  }
  previous_offset_ = record.arrival_offset_ns;
  ++index_;
  if (index_ == header_.record_count &&
      static_cast<__int128>(header_.first_timestamp_ns) + previous_offset_ !=
          header_.last_timestamp_ns) {
    throw std::runtime_error("trace timestamp mismatch");
  }
  return true;
}
} // namespace quickserve::benchmark
