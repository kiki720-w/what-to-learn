#include <argparse/argparse.hpp>
#include <lacam.hpp>
#include <memory>

static std::vector<float> load_pressure_file(const std::string& filename,
                                             const Instance& ins)
{
  auto values = std::vector<float>();
  if (filename.empty()) return values;

  auto file = std::ifstream(filename);
  if (!file) {
    std::cerr << "pressure file " << filename << " is not found" << std::endl;
    std::exit(1);
  }

  float value;
  while (file >> value) values.push_back(value);

  auto pressure = std::vector<float>(ins.G.size(), 0.0f);
  if (values.size() == ins.G.size()) {
    pressure = values;
  } else if (values.size() == ins.G.U.size()) {
    for (auto v : ins.G.V) pressure[v->id] = values[v->index];
  } else {
    std::cerr << "pressure file must contain either " << ins.G.size()
              << " vertex values or " << ins.G.U.size()
              << " grid-cell values, but got " << values.size() << std::endl;
    std::exit(1);
  }

  return pressure;
}

int main(int argc, char* argv[])
{
  // arguments parser
  argparse::ArgumentParser program("lacam", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen")
      .help("scenario file")
      .default_value(std::string(""));
  program.add_argument("-N", "--num").help("number of agents").required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .default_value(std::string("0"));
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .default_value(std::string("0"));
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .default_value(std::string("10"));
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value(std::string("./build/result.txt"));
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--pressure")
      .help("optional learned pressure file; accepts vertex values or grid-cell values")
      .default_value(std::string(""));
  program.add_argument("--pressure_weight")
      .help("weight for pressure-guided agent ordering")
      .default_value(std::string("1.0"));
  program.add_argument("--deterministic")
      .help("disable randomized neighbor shuffling and PIBT tie-breakers")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_known_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // setup instance
  const auto verbose = std::stoi(program.get<std::string>("verbose"));
  const auto time_limit_sec =
      std::stoi(program.get<std::string>("time_limit_sec"));
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = std::stoi(program.get<std::string>("seed"));
  auto MT = std::mt19937(seed);
  const auto map_name = program.get<std::string>("map");
  const auto output_name = program.get<std::string>("output");
  const auto log_short = program.get<bool>("log_short");
  const auto N = std::stoi(program.get<std::string>("num"));
  const auto pressure_name = program.get<std::string>("pressure");
  const auto pressure_weight =
      std::stof(program.get<std::string>("pressure_weight"));
  const auto deterministic = program.get<bool>("deterministic");
  std::unique_ptr<Instance> ins;
  if (scen_name.size() > 0) {
    ins = std::make_unique<Instance>(scen_name, map_name, N);
  } else {
    ins = std::make_unique<Instance>(map_name, &MT, N);
  }
  if (!ins->is_valid(1)) return 1;
  const auto pressure = load_pressure_file(pressure_name, *ins);
  const auto pressure_ptr = pressure.empty() ? nullptr : &pressure;

  // solve
  const auto deadline = Deadline(time_limit_sec * 1000);
  auto MT_ptr = deterministic ? nullptr : &MT;
  const auto solution =
      solve(*ins, verbose - 1, &deadline, MT_ptr, pressure_ptr, pressure_weight);
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) info(1, verbose, "failed to solve");

  // check feasibility
  if (!is_feasible_solution(*ins, solution, verbose)) {
    info(0, verbose, "invalid solution");
    return 1;
  }

  // post processing
  print_stats(verbose, *ins, solution, comp_time_ms);
  make_log(*ins, solution, output_name, comp_time_ms, map_name, seed, log_short);
  return 0;
}
