#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRequestMagic = 0x42515731;   // BQW1
constexpr uint32_t kResponseMagic = 0x42515231;  // BQR1
constexpr uint32_t kMsgReset = 1;
constexpr uint32_t kMsgStep = 2;

struct Args {
  std::string work_dir = "/data/local/tmp/beacon_et";
  std::string runner = "./executor_runner";
  std::string model_path = "./shard_0.pte";
  int port = 9000;
  int shard_id = 1;
  int num_layers = 0;
  int hidden_size = 3072;
  int max_cache_len = 32;
  int num_kv_heads = 8;
  int head_dim = 128;
};

struct RequestHeader {
  uint32_t magic = 0;
  uint32_t type = 0;
  uint32_t step = 0;
  uint32_t current_length = 0;
  uint32_t byte_length = 0;
};

bool read_exact(int fd, void* data, size_t size) {
  auto* cursor = static_cast<uint8_t*>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::read(fd, cursor + done, size - done);
    if (n == 0) return false;
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    done += static_cast<size_t>(n);
  }
  return true;
}

bool write_exact(int fd, const void* data, size_t size) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::write(fd, cursor + done, size - done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    done += static_cast<size_t>(n);
  }
  return true;
}

uint32_t from_be(uint32_t value) {
  return ntohl(value);
}

uint32_t to_be(uint32_t value) {
  return htonl(value);
}

std::string path_join(const std::string& a, const std::string& b) {
  if (a.empty() || a.back() == '/') return a + b;
  return a + "/" + b;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()), data.size());
  return static_cast<bool>(out);
}

bool write_zero_file(const std::string& path, size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  std::vector<uint8_t> block(8192, 0);
  size_t remaining = bytes;
  while (remaining > 0) {
    size_t n = std::min(remaining, block.size());
    out.write(reinterpret_cast<const char*>(block.data()), n);
    remaining -= n;
  }
  return static_cast<bool>(out);
}

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.seekg(0, std::ios::end);
  std::streamoff size = input.tellg();
  if (size < 0) return false;
  input.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(out->data()), out->size());
  return static_cast<bool>(input) || input.gcount() == static_cast<std::streamsize>(out->size());
}

bool copy_file(const std::string& src, const std::string& dst) {
  std::vector<uint8_t> data;
  return read_file(src, &data) && write_file(dst, data);
}

std::vector<uint8_t> int64_vector_bytes(const std::vector<int64_t>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::string shell_quote(const std::string& value) {
  std::string result = "'";
  for (char c : value) {
    if (c == '\'') {
      result += "'\\''";
    } else {
      result += c;
    }
  }
  result += "'";
  return result;
}

std::string input_path(const Args& args, int index) {
  return path_join(args.work_dir, ".beacon_in_" + std::to_string(args.shard_id) + "_" + std::to_string(index) + ".bin");
}

std::string output_base(const Args& args) {
  return path_join(args.work_dir, ".beacon_out_" + std::to_string(args.shard_id));
}

std::string output_path(const Args& args, int index) {
  return output_base(args) + "-" + std::to_string(index) + ".bin";
}

int cache_tensors(const Args& args) {
  return args.num_layers * 2;
}

size_t hidden_bytes(const Args& args) {
  return static_cast<size_t>(args.hidden_size) * sizeof(float);
}

size_t cache_bytes(const Args& args) {
  return static_cast<size_t>(args.num_kv_heads) * args.max_cache_len * args.head_dim * sizeof(float);
}

bool reset_cache(const Args& args) {
  for (int i = 0; i < cache_tensors(args); ++i) {
    if (!write_zero_file(input_path(args, 4 + i), cache_bytes(args))) return false;
  }
  return true;
}

bool prepare_inputs(const Args& args, const std::vector<uint8_t>& hidden, uint32_t step, uint32_t current_length) {
  if (hidden.size() != hidden_bytes(args)) {
    std::cerr << "hidden bytes mismatch: got " << hidden.size() << " expected " << hidden_bytes(args) << "\n";
    return false;
  }
  if (!write_file(input_path(args, 0), hidden)) return false;

  std::vector<int64_t> mask(args.max_cache_len, 0);
  for (int i = 0; i < args.max_cache_len && i < static_cast<int>(current_length); ++i) {
    mask[i] = 1;
  }
  if (!write_file(input_path(args, 1), int64_vector_bytes(mask))) return false;
  if (!write_file(input_path(args, 2), int64_vector_bytes({static_cast<int64_t>(step)}))) return false;
  if (!write_file(input_path(args, 3), int64_vector_bytes({static_cast<int64_t>(step)}))) return false;

  for (int i = 0; i < cache_tensors(args); ++i) {
    struct stat st {};
    std::string path = input_path(args, 4 + i);
    if (stat(path.c_str(), &st) != 0 || static_cast<size_t>(st.st_size) != cache_bytes(args)) {
      if (!write_zero_file(path, cache_bytes(args))) return false;
    }
  }
  return true;
}

std::string input_list(const Args& args) {
  std::ostringstream out;
  int count = 4 + cache_tensors(args);
  for (int i = 0; i < count; ++i) {
    if (i) out << ",";
    out << input_path(args, i);
  }
  return out.str();
}

bool run_executor(const Args& args) {
  std::string rm_cmd = "rm -f " + shell_quote(output_base(args)) + "-*.bin";
  std::system(rm_cmd.c_str());

  std::ostringstream cmd;
  cmd << "cd " << shell_quote(args.work_dir)
      << " && export LD_LIBRARY_PATH=" << shell_quote(args.work_dir)
      << " ADSP_LIBRARY_PATH=" << shell_quote(args.work_dir)
      << " && " << args.runner
      << " --model_path=" << shell_quote(args.model_path)
      << " --inputs=" << shell_quote(input_list(args))
      << " --print_output=none"
      << " --output_file=" << shell_quote(output_base(args))
      << " >/dev/null 2>" << shell_quote(path_join(args.work_dir, ".beacon_worker_" + std::to_string(args.shard_id) + ".log"));

  int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    std::cerr << "executor_runner failed rc=" << rc << "\n";
    return false;
  }
  for (int i = 0; i < cache_tensors(args); ++i) {
    if (!copy_file(output_path(args, 1 + i), input_path(args, 4 + i))) return false;
  }
  return true;
}

bool send_response(int fd, uint32_t status, const std::vector<uint8_t>& body) {
  uint32_t header[3] = {to_be(kResponseMagic), to_be(status), to_be(static_cast<uint32_t>(body.size()))};
  return write_exact(fd, header, sizeof(header)) && (body.empty() || write_exact(fd, body.data(), body.size()));
}

bool handle_client(int client, const Args& args) {
  reset_cache(args);
  while (true) {
    uint32_t raw[5];
    if (!read_exact(client, raw, sizeof(raw))) return true;
    RequestHeader req;
    req.magic = from_be(raw[0]);
    req.type = from_be(raw[1]);
    req.step = from_be(raw[2]);
    req.current_length = from_be(raw[3]);
    req.byte_length = from_be(raw[4]);
    if (req.magic != kRequestMagic) {
      std::string error = "bad request magic";
      send_response(client, 1, std::vector<uint8_t>(error.begin(), error.end()));
      return false;
    }
    std::vector<uint8_t> body(req.byte_length);
    if (req.byte_length && !read_exact(client, body.data(), body.size())) return false;

    if (req.type == kMsgReset) {
      bool ok = reset_cache(args);
      std::vector<uint8_t> response;
      if (!ok) {
        std::string error = "failed to reset cache files";
        response.assign(error.begin(), error.end());
      }
      send_response(client, ok ? 0 : 2, response);
      continue;
    }

    if (req.type != kMsgStep) {
      std::string error = "unknown request type";
      send_response(client, 3, std::vector<uint8_t>(error.begin(), error.end()));
      continue;
    }

    bool ok = prepare_inputs(args, body, req.step, req.current_length) && run_executor(args);
    if (!ok) {
      std::string error = "executor step failed; see .beacon_worker_" + std::to_string(args.shard_id) + ".log";
      send_response(client, 4, std::vector<uint8_t>(error.begin(), error.end()));
      continue;
    }
    std::vector<uint8_t> hidden;
    if (!read_file(output_path(args, 0), &hidden)) {
      std::string error = "failed to read hidden output";
      send_response(client, 5, std::vector<uint8_t>(error.begin(), error.end()));
      continue;
    }
    send_response(client, 0, hidden);
  }
}

int parse_int(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0') {
    std::cerr << "invalid " << name << ": " << value << "\n";
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--work_dir") args.work_dir = require_value("--work_dir");
    else if (flag == "--runner") args.runner = require_value("--runner");
    else if (flag == "--model_path") args.model_path = require_value("--model_path");
    else if (flag == "--port") args.port = parse_int(require_value("--port"), "--port");
    else if (flag == "--shard_id") args.shard_id = parse_int(require_value("--shard_id"), "--shard_id");
    else if (flag == "--num_layers") args.num_layers = parse_int(require_value("--num_layers"), "--num_layers");
    else if (flag == "--hidden_size") args.hidden_size = parse_int(require_value("--hidden_size"), "--hidden_size");
    else if (flag == "--max_cache_len") args.max_cache_len = parse_int(require_value("--max_cache_len"), "--max_cache_len");
    else if (flag == "--num_kv_heads") args.num_kv_heads = parse_int(require_value("--num_kv_heads"), "--num_kv_heads");
    else if (flag == "--head_dim") args.head_dim = parse_int(require_value("--head_dim"), "--head_dim");
    else {
      std::cerr << "unknown flag " << flag << "\n";
      std::exit(2);
    }
  }
  if (args.num_layers <= 0) {
    std::cerr << "--num_layers is required\n";
    std::exit(2);
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  reset_cache(args);

  int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    perror("socket");
    return 1;
  }
  int yes = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(args.port));
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("bind");
    return 1;
  }
  if (::listen(server, 8) != 0) {
    perror("listen");
    return 1;
  }

  std::cout << "beacon executor bridge listening on port " << args.port
            << " shard_id=" << args.shard_id
            << " model=" << args.model_path
            << " num_layers=" << args.num_layers << std::endl;

  while (true) {
    int client = ::accept(server, nullptr, nullptr);
    if (client < 0) {
      if (errno == EINTR) continue;
      perror("accept");
      return 1;
    }
    handle_client(client, args);
    ::close(client);
  }
}
