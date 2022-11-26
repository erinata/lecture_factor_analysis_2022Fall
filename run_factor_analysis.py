import pandas
from factor_analyzer import FactorAnalyzer

import numpy

dataset = pandas.read_csv("bfi_data.csv")

print(dataset)

dataset.dropna(inplace=True)

print(dataset)


machine = FactorAnalyzer(n_factors=25, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)

machine = FactorAnalyzer(n_factors=6, rotation=None)
machine.fit(dataset)
output = machine.loadings_
print(output)

machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

dataset = dataset.values



print(dataset.shape)
print(factor_loadings.shape)

results = numpy.dot(dataset, factor_loadings)

print(results)

pandas.DataFrame(results).round().to_csv("results.csv", index=False)








