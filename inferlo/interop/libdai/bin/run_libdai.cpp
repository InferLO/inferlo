#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <memory>
#include <dai/alldai.h>
#include <dai/bp.h>
#include <dai/decmap.h>
#include <dai/jtree.h>

const bool DEBUG = true;

void print_marginal_probs(const dai::FactorGraph& fg, const dai::InfAlg* inf_alg, const std::string& file_name) {
  int max_domain_size = 1;
  for (int var_id = 0; var_id < fg.nrVars(); var_id++) {
    max_domain_size = std::max(max_domain_size, (int)fg.var(var_id).states());
  }

  std::ofstream out_file(file_name);
  out_file<<std::setprecision(12);
  for(int var_id = 0; var_id < fg.nrVars(); var_id++) {
    int domain_size = fg.var(var_id).states();
    auto marg_probs = inf_alg->belief(fg.var(var_id));
    for (int j=0;j<domain_size;j++) out_file << marg_probs[j] << " ";
    for (int j=domain_size; j<max_domain_size;j++) out_file << "0 ";
    out_file << "\n";
  }
  out_file.close();

  // Write logarithm of the partition function to standard output.
  try {
    std::cout << std::setprecision(15) << inf_alg->logZ() << "\n";
  } catch (dai::Exception ex) {
    std::cout<<"0\n";
  }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cout << "Call: " + std::string(argv[0]) + " <input_file> <output_file> <problem> <algorithm> <options>\n";
        return 1;
    }
    
    std::string input_file = std::string(argv[1]);
    std::string output_file = std::string(argv[2]);
    std::string problem = std::string(argv[3]);
    std::string algorithm_name = std::string(argv[4]);
    dai::PropertySet opts = dai::PropertySet(std::string(argv[5]));
    

    // Read the model from file.
    dai::FactorGraph fg;
    fg.ReadFromFile(input_file.c_str());
    
    // Create InfAlg object.
    dai::InfAlg* inf_alg = dai::newInfAlg(algorithm_name, fg, opts);
    if (DEBUG) {((dai::BP*)inf_alg)->recordSentMessages = true;}
    inf_alg->init();
    inf_alg->run();

    if (problem =="infer") {
        print_marginal_probs(fg, inf_alg, output_file);
    } else if (problem == "max_likelihood") {
        std::ofstream out_file(output_file);
        for (auto x: inf_alg->findMaximum()) {out_file << x << " ";}
    } else {
        std::cerr << "Unknown problem " << problem << "\n";
        return 1;
    }

    if (DEBUG) {
        dai::BP bp = *((dai::BP*)inf_alg);
        std::cerr << "Sent messages:\n";
        for(const std::pair<std::size_t, std::size_t>& msg: bp.getSentMessages()) {
            std::cerr << msg.first << " " << msg.second << "\n";
        }

        std::cerr << "Beliefs:\n";
        for(int i=0; i < bp.nrVars(); i++) {
            std::cerr << "beliefV(" << i << ")=" << bp.beliefV(i) << "\n";
        }
        for(int i=0; i < bp.nrFactors(); i++) {
            dai::Factor b = bp.beliefF(i);
            std::cerr << "beliefF(" << i << ")=" << b << "\n";
        }

        std::cerr << "Factors:\n";
        for(int i=0; i < bp.nrFactors(); i++) {
            std::cerr << "factor(" << i << ")=" << bp.factor(i) << "\n";
        }
    }

    return 0;
}
