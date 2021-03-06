# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from inferlo import GenericGraphModel


class UaiWriter():
    """Writes Graphical Model in UAI format."""

    def write_model(self, model: GenericGraphModel, path: str):
        """Writes Graphical model to a file in UAI format.

        Format description:
        http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html

        :param model: Graphical model to write.
        :param path: Local path to text file with GM in UAI format.
        """
        assert path.endswith('.uai')
        with open(path, 'w') as f:
            # Write model type.
            f.write('MARKOV\n')

            # Write variables' cardinalities and initialize the model.
            f.write(str(model.num_variables) + '\n')
            for var_id in range(model.num_variables):
                card = model.get_variable(var_id).domain.size()
                assert 1 <= card <= 1000
                f.write(str(card) + ' ')
            f.write('\n')

            # Write which variables are in which factors.
            num_factors = len(model.factors)
            f.write(str(num_factors) + '\n')
            for factor_id in range(num_factors):
                var_idx = model.factors[factor_id].var_idx
                f.write(str(len(var_idx)) + ' ')
                f.write(' '.join(str(var_id) for var_id in var_idx))
                f.write('\n')

            # Write factors' values.
            for factor_id in range(num_factors):
                factor = model.factors[factor_id]
                assert factor.is_discrete()
                vals_flat = factor.values.reshape(-1)
                f.write(str(len(vals_flat)))
                f.write('\n')
                f.write(' '.join(['%.10f' % x for x in vals_flat]))
                f.write('\n')

            return model
