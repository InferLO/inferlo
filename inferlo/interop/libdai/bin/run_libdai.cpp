#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <dai/alldai.h>
#include <dai/jtree.h>
#include <dai/bp.h>
#include <dai/decmap.h>

void print_marginal_probs(const dai::FactorGraph& fg, const dai::BP& bp, std::string file_name) {
  int max_domain_size = 1;
  for (int var_id = 0; var_id < fg.nrVars(); var_id++) {
    max_domain_size = std::max(max_domain_size, (int)fg.var(var_id).states());
  }

  std::ofstream out_file(file_name);
  out_file<<std::setprecision(12);
  for(int var_id = 0; var_id < fg.nrVars(); var_id++) {
    int domain_size = fg.var(var_id).states();
    auto marg_probs = bp.belief(fg.var(var_id));
    for (int j=0;j<domain_size;j++) out_file << marg_probs[j] << " ";
    for (int j=domain_size; j<max_domain_size;j++) out_file << "0 ";
    out_file << "\n";
  }
  out_file.close();

  // Write logarithm of the partition function to standard output.
  std::cout << std::setprecision(15) << bp.logZ() << "\n";
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Call: " + std::string(argv[0]) + " <input_file> <output_file> <problem> <algorithm>\n";
        return 1;
    }
    std::string input_file = std::string(argv[1]);
    std::string output_file = std::string(argv[2]);
    std::string problem = std::string(argv[3]);
    std::string algorithm = std::string(argv[4]);

    dai::FactorGraph fg;
    fg.ReadFromFile(input_file.c_str());

    // Set some constants
    size_t maxiter = 10000;
    dai::Real tol = 1e-9;
    size_t verb = 0;

    // Store the constants in a PropertySet object
    dai::PropertySet opts;
    opts.set("maxiter", maxiter);  // Maximum number of iterations
    opts.set("tol", tol);          // Tolerance for convergence
    opts.set("verbose", verb);     // Verbosity (amount of output generated)

    if (problem =="infer") {
        // Loopy belief propagation.
        dai::BP bp(fg, opts("updates",std::string("SEQRND"))("logdomain",false));
        bp.init();
        bp.run();
        print_marginal_probs(fg, bp, output_file);
    } else if (problem == "max_likelihood") {
        // Max-Product algorithm.
        dai::BP mp(fg, opts("updates",std::string("SEQRND"))
                           ("logdomain",false)
                           ("inference",std::string("MAXPROD"))
                           ("damping",std::string("0.1")));
        mp.init();
        mp.run();
        std::ofstream out_file(output_file);
        for (auto x: mp.findMaximum()) {out_file << x << " ";}
    } else {
        std::cout << "Unknown problem " << problem << "\n";
        return 1;
    }
    return 0;
}
